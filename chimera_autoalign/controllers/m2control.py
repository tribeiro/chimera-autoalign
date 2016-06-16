from __future__ import division

from chimera.core.chimeraobject import ChimeraObject
from chimera.core.lock import lock
from chimera.core.event import event
from chimera.core.exceptions import ChimeraException, ClassLoaderException
from chimera.interfaces.focuser import InvalidFocusPositionException, FocuserAxis
from chimera.util.coord import Coord
from chimera.util.position import Position

from chimera_autoalign.util.mkoptics import HexapodAxes

from astropy.io import fits
from astropy import units
from astropy.coordinates import Angle
import numpy as np
import threading

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
                    self.update()
                except Exception, e:
                    self.log.exception(e)

            self.loopControl.release()

        self.workingthread = threading.Thread(target=worker)

        self.workingthread.setDaemon(True)

        self.workingthread.start()


    def __stop__(self):

        self._disconnectTelescopeEvents()

    def loadLookupTable(self, tablename, hduindex = 1):

        hdulist = fits.open(tablename)

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

        self.lookuptable = hdulist[hduindex].data

        # TOdo: Check dtype of loopuptable

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

        newhdu.writeto(tablename)

    def getLookupTable(self):
        return self.lookuptable

    @lock
    def activate(self):

        self.loopControl.acquire()
        self.loop_timeout = self["update_time"]
        self.loopControl.notify()
        self.loopControl.release()

    @lock
    def deactivate(self):
        self.loopControl.acquire()
        self.loop_timeout = None
        self.loopControl.release()

    @lock
    def update(self):

        self.log.debug("Updating M2 position...")

        tel = self.getTel()
        focuser = self.getFocuser()

        pos = tel.getPositionAltAz()

        dist = np.array([
                            pos.angsep(Position.fromAltAz(
                                Coord.fromD(self.lookuptable['ALT'][i]),
                                Coord.fromD(self.lookuptable['AZ'][i]))).D for i in range(len(self.lookuptable))
                         ]
                        )

        if self['distance_tolerance'] is not None and np.min(dist) > self["distance_tolerance"]:
            self.log.debug('Current position too far from closest value')
            return

        index = np.argmin(dist)

        offset_list = [
            (self.lookuptable['X'][index] - self.lookuptable['RX'][index] - self.refOffset.x.to(units.mm).value,
             FocuserAxis.X, 1e-3),
            (self.lookuptable['Y'][index] - self.lookuptable['RY'][index] - self.refOffset.y.to(units.mm).value,
             FocuserAxis.Y, 1e-3),
            (self.lookuptable['Z'][index] - self.lookuptable['RZ'][index] - self.refOffset.z.to(units.mm).value,
             FocuserAxis.Z, 1e-3),
            (self.lookuptable['U'][index] - self.lookuptable['RU'][index] - self.refOffset.u.to(units.degree).value,
             FocuserAxis.U, 1e-3),
            (self.lookuptable['V'][index] - self.lookuptable['RV'][index] - self.refOffset.v.to(units.degree).value,
             FocuserAxis.V, 1e-3), ]

        for offset in offset_list:
            if offset[0] > offset[2]:
                focuser.moveOut(abs(offset[0]),axis=offset[1])
            elif -offset[0] > offset[2]:
                focuser.moveIn(abs(offset[0]),axis=offset[1])

        self.updateComplete(offset_list)

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

    def setRefOffset(self):
        '''
        Sets reference offset to current focuser position.
        :return:
        '''

        focuser = self.getFocuser()
        if focuser is None:
            self.log.warning("Couldn't find focuser.")
            return False

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

    def _connectTelescopeEvents(self):
        # Todo
        tel = self.getTel()
        if tel is None:
            self.log.warning("Couldn't find telescope.")
            return False

        tel.slewBegin += self.getProxy()._watchSlewBegin
        tel.slewComplete += self.getProxy()._watchSlewComplete
        tel.trackingStarted += self.getProxy()._watchTrackingStarted
        tel.trackingStopped += self.getProxy()._watchTrackingStopped

        return True

    def _disconnectTelescopeEvents(self):
        tel = self.getTel()
        if tel is None:
            self.log.warning("Couldn't find telescope.")
            return False

        tel.slewBegin -= self.getProxy()._watchSlewBegin
        tel.trackingStarted -= self.getProxy()._watchTrackingStarted
        tel.trackingStopped -= self.getProxy()._watchTrackingStopped

    def _watchSlewBegin(self):

        self.abort.set()
        self.loop_timeout = None

    def _watchTrackingStarted(self):

        if self["auto"]:
            self.abort.clear()
            self.loop_timeout = self["update_time"]

    def _watchTrackingStopped(self):

        self.abort.set()
        self.loop_timeout = None

    @event
    def updateComplete(self, position):
        """
        """
        pass