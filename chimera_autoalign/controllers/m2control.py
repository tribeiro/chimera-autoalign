from __future__ import division

from chimera.core.chimeraobject import ChimeraObject
from chimera.core.lock import lock
from chimera.core.event import event
from chimera.core.exceptions import ChimeraException, ClassLoaderException
from chimera.interfaces.focuser import InvalidFocusPositionException, FocuserAxis
from chimera.util.coord import Coord
from chimera.util.position import Position
from chimera.util.enum import Enum

from chimera_autoalign.util.mkoptics import HexapodAxes

from collections import OrderedDict

from astropy.io import fits
from astropy import units
from astropy.coordinates import Angle
import numpy as np
import threading

State = Enum("ACTIVE", "STOP", "ERROR", "RANGE")

interpolationType = ['nearest', 'linear']

lookuptable_dtype = [('NAME', 'a25'), ('ALT', np.float), ('AZ', np.float),
                     ('TM1', np.float), ('TM2', np.float), ('FrontRing', np.float), ('TubeRod', np.float),
                     ('X', np.float), ('Y', np.float), ('Z', np.float),
                     ('U', np.float), ('V', np.float),
                     ('DX', np.float), ('DY', np.float), ('DZ', np.float),
                     ('DU', np.float), ('DV', np.float),
                     ('RX', np.float), ('RY', np.float), ('RZ', np.float),
                     ('RU', np.float), ('RV', np.float),
                     ('MASK_DX', np.int), ('MASK_DY', np.int), ('MASK_DZ', np.int),
                     ('MASK_DU', np.int), ('MASK_DV', np.int), ('MASK_T', np.int)]

class LookupTableException(ChimeraException):
    pass

class M2Model():

    CX = np.array([])
    CY = np.array([])
    CZ = np.array([])
    CU = np.array([])
    CV = np.array([])

    def __getitem__(self, item):
        if item == 'X' or item == 'CX':
            return self.CX
        elif item == 'Y' or item == 'CY':
            return self.CY
        elif item == 'Z' or item == 'CZ':
            return self.CZ
        elif item == 'U' or item == 'CU':
            return self.CU
        elif item == 'V' or item == 'CV':
            return self.CV
        else:
            raise AttributeError('No %s choose between X, Y, Z, U or V' % item)

    def __setitem__(self, key, value):

        if key == 'X' or key == 'CX':
            self.CX = value
        elif key == 'Y' or key == 'CY':
            self.CY = value
        elif key == 'Z' or key == 'CZ':
            self.CZ = value
        elif key == 'U' or key == 'CU':
            self.CU = value
        elif key == 'V' or key == 'CV':
            self.CV = value
        else:
            raise AttributeError('No %s choose between X, Y, Z, U or V' % item)

class M2Control(ChimeraObject):
    """
    M2 Control
    ==========

    This controller implement lookup table for controlling the M2 position usefull for hexapod based
    secondary or temperature control focusers.

    It reads a fits table with focuser information and will use a linear interpolation over a sphere between the 4
    nearest points to determine the focuser position. It will update the position every ``update_time`` seconds.
    The controller will start/stop with telescope tracking events.

    """

    __config__ = {'table' : None,    # a FITS file with the lookup table data. See help for the format of the table.
                  'focuser' : None,
                  'telescope' : None, # need a telescope in order to watch for slew/tracking events.
                  'auto'    : False, # automatically start/stop controller on telescope events?
                  'update_time' : 120 ,
                  'temperature_compensation' : False,
                  'temperature_column_name' : None ,
                  'interpolation' : 'nearest' ,
                  'distance_tolerance' : None, # Tolerance in the distance between nearest points and current position.
    }

    def __init__(self):

        ChimeraObject.__init__(self)

        self.loopControl = threading.Condition()
        self.loop_timeout = None
        self.abort = threading.Event()
        self.workingthread = None

        self.lookuptable = None
        self.refPos = HexapodAxes()
        self.refOffset = HexapodAxes()
        self.currentPos = HexapodAxes()
        self.align_focus = None
        self.state = State.STOP
        self.m2coef = M2Model()

    def __start__(self):

        if self["table"] is not None:
            self.loadLookupTable(self["table"]) # will check if table contains everything it needs to
        else:
            self.lookuptable = np.array([],dtype=lookuptable_dtype)

        self._connectTelescopeEvents()

        def worker():


            self.loopControl.acquire()
            while True:

                try:
                    self.loopControl.wait(timeout=self.loop_timeout)
                    if self.state == State.ACTIVE:
                        self.update()
                    else:
                        self.log.warning("Current state is %s, loop will pause." % self.state)
                        self.loop_timeout = None
                except Exception, e:
                    self.state = State.ERROR
                    self.log.exception(e)

            self.loopControl.release()

        self.workingthread = threading.Thread(target=worker)

        self.workingthread.setDaemon(True)

        self.workingthread.start()


    def __stop__(self):

        self._disconnectTelescopeEvents()

    def loadLookupTable(self, tablename, hduindex = 1):

        hdulist = fits.open(tablename)

        try:
            self.refPos.x = hdulist[hduindex].header['REFX'] * units.mm
            self.refPos.y = hdulist[hduindex].header['REFY'] * units.mm
            self.refPos.z = hdulist[hduindex].header['REFZ'] * units.mm
            self.refPos.u = hdulist[hduindex].header['REFU'] * units.degree
            self.refPos.v = hdulist[hduindex].header['REFV'] * units.degree

            self.refOffset.x = hdulist[hduindex].header['OREFX'] * units.mm
            self.refOffset.y = hdulist[hduindex].header['OREFY'] * units.mm
            self.refOffset.z = hdulist[hduindex].header['OREFZ'] * units.mm
            self.refOffset.u = hdulist[hduindex].header['OREFU'] * units.degree
            self.refOffset.v = hdulist[hduindex].header['OREFV'] * units.degree

            if 'ALFOC' in hdulist[hduindex].header.keys():
                self.align_focus = hdulist[hduindex].header['ALFOC']


            self.m2coef.CX = np.zeros(hdulist[hduindex]['NCX'])
            self.m2coef.CY = np.zeros(hdulist[hduindex]['NCY'])
            self.m2coef.CZ = np.zeros(hdulist[hduindex]['NCZ'])
            self.m2coef.CU = np.zeros(hdulist[hduindex]['NCU'])
            self.m2coef.CV = np.zeros(hdulist[hduindex]['NCV'])

            for i in range(len(self.m2coef.CX)):
                self.m2coef.CX[i] = hdulist[hduindex]['CX%i' % i]
            for i in range(len(self.m2coef.CY)):
                self.m2coef.CY[i] = hdulist[hduindex]['CY%i' % i]
            for i in range(len(self.m2coef.CZ)):
                self.m2coef.CZ[i] = hdulist[hduindex]['CZ%i' % i]
            for i in range(len(self.m2coef.CU)):
                self.m2coef.CU[i] = hdulist[hduindex]['CU%i' % i]
            for i in range(len(self.m2coef.CV)):
                self.m2coef.CV[i] = hdulist[hduindex]['CV%i' % i]
        except Exception, e:
            self.log.warning('Could not load model coeficients. Fitting new table.')
            self.fitM2Control()
            

        self.lookuptable = hdulist[hduindex].data

        # TODO: Check dtype of lookuptable

    def saveLookupTable(self, tablename):

        newhdu = fits.BinTableHDU(data = self.lookuptable)
        newhdu.header['REFX'] = self.refPos.x.to(units.mm).value
        newhdu.header['REFY'] = self.refPos.y.to(units.mm).value
        newhdu.header['REFZ'] = self.refPos.z.to(units.mm).value
        newhdu.header['REFU'] = self.refPos.u.to(units.degree).value
        newhdu.header['REFV'] = self.refPos.v.to(units.degree).value

        newhdu.header['OREFX'] = self.refOffset.x.to(units.mm).value
        newhdu.header['OREFY'] = self.refOffset.y.to(units.mm).value
        newhdu.header['OREFZ'] = self.refOffset.z.to(units.mm).value
        newhdu.header['OREFU'] = self.refOffset.u.to(units.degree).value
        newhdu.header['OREFV'] = self.refOffset.v.to(units.degree).value

        newhdu.header['ALFOC'] = self.align_focus

        newhdu.header['NCX'] = len(self.m2coef.CX)
        newhdu.header['NCY'] = len(self.m2coef.CY)
        newhdu.header['NCZ'] = len(self.m2coef.CZ)
        newhdu.header['NCU'] = len(self.m2coef.CU)
        newhdu.header['NCV'] = len(self.m2coef.CV)

        for i,c in enumerate(self.m2coef.CX):
            newhdu.header['CX%i' % i] = c
        for i,c in enumerate(self.m2coef.CY):
            newhdu.header['CY%i' % i] = c
        for i,c in enumerate(self.m2coef.CZ):
            newhdu.header['CZ%i' % i] = c
        for i,c in enumerate(self.m2coef.CU):
            newhdu.header['CU%i' % i] = c
        for i,c in enumerate(self.m2coef.CV):
            newhdu.header['CV%i' % i] = c

        newhdu.writeto(tablename)

    def getLookupTable(self):
        return self.lookuptable


    def getState(self):
        self.loopControl.acquire()
        rstate = self.state
        self.loopControl.release()

        return rstate

    @lock
    def activate(self):

        self.loopControl.acquire()
        self.state = State.ACTIVE
        self.abort.clear()
        self.loop_timeout = self["update_time"]
        self.loopControl.notify()
        self.loopControl.release()

    @lock
    def deactivate(self):
        self.log.debug('Deactivating m2 control law loop...')
        self.loopControl.acquire()
        self.state = State.STOP
        self.abort.set()
        self.loop_timeout = None
        self.loopControl.release()

    @lock
    def update(self, auto=False):

        self.log.debug("Updating M2 position...")

        focuser = self.getFocuser()

        self.log.debug('Getting model position...')

        position_list = self.getModelPosition()

        #
        # max_move_factor = 5.
        #
        self.abort.clear()
        for position in position_list:
            self.log.debug('Updating %s' % position[1])
            currentpos = focuser.getPosition(position[1])
            self.log.debug('Moving %s to %6.3f (current position is %6.3f)' % (position[1],position[0],currentpos))
            if self.abort.isSet():
                self.log.debug('M2 control sequence aborted! Stop!')
                break
            elif np.abs(position[0] - currentpos) > 1e-3: # FIXME: Hard coded tolerance!!!
                focuser.moveTo(position[0]/focuser[position[2]],axis=position[1])
            else:
                self.log.debug('Offset in %s too small... Skipping...' % position[1])
        #
        #     currentpos = focuser.getPosition(offset[1])
        #     diff = offset[0] - currentpos
        #     if (offset[2] <  np.abs(diff) < max_move_factor*offset[2]) or \
        #             (np.abs(diff) > max_move_factor*offset[2] and self.state != State.ACTIVE):
        #         self.log.debug('Moving %s to %6.3f (current position is %6.3f)' % (offset[1],offset[0],currentpos))
        #         focuser.moveTo(offset[0]/focuser[offset[3]],axis=offset[1])
        #     elif np.abs(diff) >= max_move_factor*offset[2]:
        #         moveto = currentpos +  max_move_factor*offset[2] if diff > 0 else currentpos - max_move_factor*offset[2]
        #         self.log.debug('Offset on %s too large (%+6.2e). Maximum %5.2e. '
        #                        'Moving to %6.3f.' % (offset[1],
        #                                              np.abs(diff),
        #                                              max_move_factor * offset[2],
        #                                              moveto))
        #         focuser.moveTo(moveto/focuser[offset[3]],axis=offset[1])
        #     else:
        #         self.log.debug('Offset in %s (%+6.1e) too small (threshold = %5.2e).' % (offset[1],
        #                                                                                 diff,
        #                                                                                 offset[2]))


        self.updateComplete(position_list)

    def getTel(self):
        if self["telescope"] is not None:
            return self.getManager().getProxy(self["telescope"])
        else:
            return None

    def getFocuser(self):
        if self["focuser"] is not None:
            return self.getManager().getProxy(self["focuser"])
        else:
            return None

    def setRefPos(self):
        '''
        Sets reference position to current focuser position.
        :return:
        '''

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        self.refPos.x = focuser.getPosition(FocuserAxis.X)*units.mm
        self.refPos.y = focuser.getPosition(FocuserAxis.Y)*units.mm
        self.refPos.z = focuser.getPosition(FocuserAxis.Z)*units.mm
        self.refPos.u = focuser.getPosition(FocuserAxis.U)*units.degree
        self.refPos.v = focuser.getPosition(FocuserAxis.V)*units.degree

    def getRefPos(self):
        return self.refPos

    def getCurrentSavedPos(self):
        return self.currentPos

    def getRefOffset(self):
        return self.refOffset

    def setRefOffset(self):
        '''
        Sets reference offset to current focuser position.
        :return:
        '''

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        self.log.debug('%s %s' % (self.refPos.x,focuser.getPosition(FocuserAxis.X)*units.mm))
        self.refOffset.x = self.refPos.x-focuser.getPosition(FocuserAxis.X)*units.mm
        self.refOffset.y = self.refPos.y-focuser.getPosition(FocuserAxis.Y)*units.mm
        self.refOffset.z = self.refPos.z-focuser.getPosition(FocuserAxis.Z)*units.mm
        self.refOffset.u = self.refPos.v-focuser.getPosition(FocuserAxis.U)*units.degree
        self.refOffset.v = self.refPos.w-focuser.getPosition(FocuserAxis.V)*units.degree

    def goRefPos(self):

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        focuser.moveTo(self.refPos.x.to(units.mm).value/focuser['step_x'],FocuserAxis.X)
        focuser.moveTo(self.refPos.y.to(units.mm).value/focuser['step_y'],FocuserAxis.Y)
        focuser.moveTo(self.refPos.z.to(units.mm).value/focuser['step_z'],FocuserAxis.Z)
        focuser.moveTo(self.refPos.u.to(units.degree).value/focuser['step_u'],FocuserAxis.U)
        focuser.moveTo(self.refPos.v.to(units.degree).value/focuser['step_v'],FocuserAxis.V)

    def savePosition(self):
        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        self.currentPos.x = focuser.getPosition(FocuserAxis.X)*units.mm
        self.currentPos.y = focuser.getPosition(FocuserAxis.Y)*units.mm
        self.currentPos.z = focuser.getPosition(FocuserAxis.Z)*units.mm
        self.currentPos.u = focuser.getPosition(FocuserAxis.U)*units.degree
        self.currentPos.v = focuser.getPosition(FocuserAxis.V)*units.degree

    def setupOffset(self):
        '''
        Will setup the current offset based on the saved current position and the actuall current position of the
        hexapod
        :return:
        '''

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        # offset_list = self.getOffset()
        self.refOffset.x = focuser.getPosition(FocuserAxis.X)*units.mm-self.currentPos.x
        self.refOffset.y = focuser.getPosition(FocuserAxis.Y)*units.mm-self.currentPos.y
        self.refOffset.z = focuser.getPosition(FocuserAxis.Z)*units.mm-self.currentPos.z
        self.refOffset.u = focuser.getPosition(FocuserAxis.U)*units.degree-self.currentPos.u
        self.refOffset.v = focuser.getPosition(FocuserAxis.V)*units.degree-self.currentPos.v

    def reset(self):
        self.currentPos = HexapodAxes()
        self.refOffset = HexapodAxes()

    def calibrate(self):

        self.reset()
        self.setRefPos()

        offset_list = self.getModelPosition()
        self.currentPos.x = offset_list[0][0]*units.mm
        self.currentPos.y = offset_list[1][0]*units.mm
        self.currentPos.z = offset_list[2][0]*units.mm
        self.currentPos.u = offset_list[3][0]*units.degree
        self.currentPos.v = offset_list[4][0]*units.degree

        self.setupOffset()


    def add(self,name=''):

        tel = self.getTel()
        if tel is None:
            self.log.warning("Couldn't find telescope.")
            return False

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

        temp = tel.getSensors()

        tdict = {}

        for entry in temp:
            tdict[entry[0]] = entry[1]

        entry = np.array([(name, tel.getAlt(), tel.getAz(),
                           tdict['TM1'], tdict['TM2'], tdict['FrontRing'], tdict['TubeRod'],
                           focuser.getPosition(FocuserAxis.X),
                           focuser.getPosition(FocuserAxis.Y),
                           focuser.getPosition(FocuserAxis.Z),
                           focuser.getPosition(FocuserAxis.U),
                           focuser.getPosition(FocuserAxis.V),
                           focuser.getOffset(FocuserAxis.X),
                           focuser.getOffset(FocuserAxis.Y),
                           focuser.getOffset(FocuserAxis.Z),
                           focuser.getOffset(FocuserAxis.U),
                           focuser.getOffset(FocuserAxis.V),
                           self.refPos.x.to(units.mm).value,
                           self.refPos.y.to(units.mm).value,
                           self.refPos.z.to(units.mm).value,
                           self.refPos.u.to(units.degree).value,
                           self.refPos.v.to(units.degree).value,
                           0, 0, 0, 0, 0, 0),
                          ], dtype=lookuptable_dtype)

        self.lookuptable = np.append(self.lookuptable,entry)

    def getOffset(self):

        tel = self.getTel()
        pos = tel.getPositionAltAz()

        dist = np.array([
                            pos.angsep(Position.fromAltAz(
                                Coord.fromD(self.lookuptable['ALT'][i]),
                                Coord.fromD(self.lookuptable['AZ'][i]))).D for i in range(len(self.lookuptable))
                         ]
                        )

        if self['distance_tolerance'] is not None and np.min(dist) > self["distance_tolerance"]:
            self.log.warning('Current position too far from closest value')
            self.state = State.RANGE

        index = np.argmin(dist)

        return [
            (self.lookuptable['X'][index] - self.lookuptable['RX'][index] + self.refPos.x.to(units.mm).value -
             self.refOffset.x.to(units.mm).value,
              FocuserAxis.X, 5e-4,'step_x'),
            (self.lookuptable['Y'][index] - self.lookuptable['RY'][index] + self.refPos.y.to(units.mm).value -
             self.refOffset.y.to(units.mm).value,
             FocuserAxis.Y, 5e-4,'step_y'),
            (self.lookuptable['Z'][index] - self.lookuptable['RZ'][index] + self.refPos.z.to(units.mm).value -
             self.refOffset.z.to(units.mm).value,
             FocuserAxis.Z, 5e-4,'step_z'),
            (self.lookuptable['U'][index] - self.lookuptable['RU'][index] + self.refPos.u.to(units.degree).value -
             self.refOffset.u.to(units.degree).value,
             FocuserAxis.U, 1e-5,'step_u'),
            (self.lookuptable['V'][index] - self.lookuptable['RV'][index] + self.refPos.v.to(units.degree).value -
             self.refOffset.v.to(units.degree).value,
             FocuserAxis.V, 1e-5,'step_v'), ]

        # return [
        #     (self.lookuptable['X'][index] - self.refOffset.x.to(units.mm).value +
        #      self.lookuptable['RX'][index] - self.refPos.x.to(units.mm).value,
        #      FocuserAxis.X, 1e-3,'step_x'),
        #     (self.lookuptable['Y'][index] - self.refOffset.y.to(units.mm).value +
        #      self.lookuptable['RY'][index] - self.refPos.y.to(units.mm).value,
        #      FocuserAxis.Y, 1e-3,'step_y'),
        #     (self.lookuptable['Z'][index] - self.refOffset.z.to(units.mm).value +
        #      self.lookuptable['RZ'][index] - self.refPos.z.to(units.mm).value,
        #      FocuserAxis.Z, 1e-3,'step_z'),
        #     (self.lookuptable['U'][index] - self.refOffset.u.to(units.degree).value +
        #      self.lookuptable['RU'][index] - self.refPos.u.to(units.degree).value,
        #      FocuserAxis.U, 1e-3,'step_u'),
        #     (self.lookuptable['V'][index] - self.refOffset.v.to(units.degree).value +
        #      self.lookuptable['RV'][index] - self.refPos.v.to(units.degree).value,
        #      FocuserAxis.V, 1e-3,'step_v'), ]

    def getAlignFocus(self):
        return self.align_focus

    def setAlignFocus(self,value):
        self.align_focus = value

    def getModelPosition(self):

        tel = self.getTel()
        pos = tel.getPositionAltAz()

        alt = pos.alt.R
        az = pos.az.R

        temp = tel.getSensors()

        tdict = {}

        for entry in temp:
            tdict[entry[0]] = entry[1]

        GX = np.cos(alt)*np.sin(az)
        GY = np.cos(alt)*np.cos(az)
        GZ = np.sin(alt)
        T = tdict['TubeRod']

        def fitfunc(p,coords):
            x,y,z,t = coords[0],coords[1],coords[2],coords[3]
            return p[0] + p[1] * x + p[2] * y + p[3] * z + p[4] * t + p[5] * t * t

        return [(fitfunc(self.m2coef['X'],[GX,GY,GZ,T]) + self.refOffset.x.to(units.mm).value,
                 FocuserAxis.X, 'step_x'),
                (fitfunc(self.m2coef['Y'],[GX,GY,GZ,T]) + self.refOffset.y.to(units.mm).value,
                 FocuserAxis.Y, 'step_y'),
                (fitfunc(self.m2coef['Z'],[GX,GY,GZ,T]) + self.refOffset.z.to(units.mm).value,
                 FocuserAxis.Z, 'step_z'),
                (fitfunc(self.m2coef['U'],[GX,GY,GZ,T]) + self.refOffset.u.to(units.degree).value,
                 FocuserAxis.U, 'step_u'),
                (fitfunc(self.m2coef['V'],[GX,GY,GZ,T]) + self.refOffset.v.to(units.degree).value,
                FocuserAxis.V, 'step_v'),
                ]

    def fitM2Control(self):

        for axis in 'XYZUV':
            self._fitM2ControlAxis(axis)

        return True

    def _fitM2ControlAxis(self,axis):
        '''
        Fit M2 control law for a specific axis.
        :param axis:
        :param orders:
        :return:
        '''

        GX = np.cos(self.lookuptable['ALT'] * np.pi / 180.)*np.sin(self.lookuptable['AZ'] * np.pi / 180.)
        GY = np.cos(self.lookuptable['ALT'] * np.pi / 180.)*np.cos(self.lookuptable['AZ'] * np.pi / 180.)
        GZ = np.sin(self.lookuptable['ALT'] * np.pi / 180.)
        T = self.lookuptable['TubeRod']

        data = self.lookuptable[axis]

        from scipy.optimize import leastsq


        def fitfunc(p,coords):
            x,y,z,t = coords[0],coords[1],coords[2],coords[3]
            return p[0] + p[1] * x + p[2] * y + p[3] * z + p[4] * t + p[5] * t * t

        errfunc = lambda p, x: fitfunc(p, x) - x[4]

        p0 = np.zeros(6)+1.e-1

        # Get good starting points for zero-point plus temperature dependency
        ct = np.polyfit(T,data,2)

        p0[0] = ct[0]
        p0[-2] = ct[1]
        p0[-1] = ct[2]

        p1, flag = leastsq(errfunc, p0, args=([GX,GY,GZ,T,data],))

        self.m2coef[axis] = p1

        return flag

    def _connectTelescopeEvents(self):
        # Todo
        tel = self.getTel()
        if tel is None:
            self.log.warning("Couldn't find telescope.")
            return False

        tel.slewBegin += self.getProxy()._watchSlewBegin
        # tel.slewComplete += self.getProxy()._watchSlewComplete
        tel.trackingStarted += self.getProxy()._watchTrackingStarted
        tel.trackingStopped += self.getProxy()._watchTrackingStopped

        return True

    def _disconnectTelescopeEvents(self):
        tel = self.getTel()
        if tel is None:
            self.log.warning("Couldn't find telescope.")
            return False

        tel.slewBegin -= self.getProxy()._watchSlewBegin
        # tel.slewComplete -= self.getProxy()._watchSlewComplete
        tel.trackingStarted -= self.getProxy()._watchTrackingStarted
        tel.trackingStopped -= self.getProxy()._watchTrackingStopped

    def _watchSlewBegin(self,target):

        # self.abort.set()
        # self.loop_timeout = None
        self.deactivate()

    def _watchTrackingStarted(self, position):

        if self["auto"]:
            self.activate()
            # self.abort.clear()
            # self.loop_timeout = self["update_time"]

    def _watchTrackingStopped(self, position, status):

        self.deactivate()
        # self.abort.set()
        # self.loop_timeout = None

    @event
    def updateComplete(self, position):
        """
        """
        pass
