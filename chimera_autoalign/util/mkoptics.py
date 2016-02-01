
from subprocess import Popen,PIPE,STDOUT
from chimera.util.sextractor import SExtractor
from astropy import units
import re
import json
from tempfile import NamedTemporaryFile
from astropy.io import fits
from donut.zernmap import ZernMap
import numpy as np
import os
import shutil

import logging
#import chimera.core.log
log = logging.getLogger(__name__.replace('chimera_autoalign','chimera'))

class MkOpticsException(Exception):
    pass

class HexapodAxes():
    
    _x = 0.*units.meter
    _y = 0.*units.meter
    _z = 0.*units.meter
    _u = 0.*units.degree
    _v = 0.*units.degree
    _w = 0.*units.degree

    _seeing = 0.*units.arcsec

    @units.quantity_input(x=units.meter)
    def setX(self,x):
        self._x = x
    @units.quantity_input(y=units.meter)
    def setY(self,y):
        self._y = y
    @units.quantity_input(z=units.meter)
    def setZ(self,z):
        self._z = z
    @units.quantity_input(u=units.degree)
    def setU(self,u):
        self._u = u
    @units.quantity_input(v=units.degree)
    def setV(self,v):
        self._v = v
    @units.quantity_input(w=units.degree)
    def setW(self,w):
        self._w = w
    @units.quantity_input(s=units.arcsec)
    def setSeeing(self,s):
        self._s = s

    def getX(self):
        return self._x
    def getY(self):
        return self._y
    def getZ(self):
        return self._z
    def getU(self):
        return self._u
    def getV(self):
        return self._v
    def getW(self):
        return self._w
    def getSeeing(self):
        return self._s

    x = property(getX,setX)
    y = property(getY,setY)
    z = property(getZ,setX)
    u = property(getU,setU)
    v = property(getV,setV)
    w = property(getW,setW)
    seeing = property(getSeeing,setSeeing)

    def __getitem__(self, item):
        if item == 'comma':
            return {'X':self.x,'Y':self.y}
        elif item == 'astigmatism':
            return {'U':self.u,'V':self.v}
        elif item == 'focus':
            return {'Z':self.z}
        else:
            raise IndexError('Options are: comma, astigmatism or focus.')

    def __setitem__(self, key, value):
        if key == 'comma':
            self.x,self.y = value[0],value[1]
        elif key == 'astigmatism':
            self.u,self.v = value[0],value[1]
        elif key == 'focus':
            self.z = value
        else:
            raise IndexError('Options are: comma, astigmatism or focus. Got %s' % key)

class MkOptics(SExtractor):

    ZERNPAR = { "donpar":[ {    "D": 0.8200  ,
                               "EPS": 0.322 ,
                               "ALAMBDA": 0.800000 ,
                               "PIXEL": 0.55000 ,
                               "NGRID": 64 ,
                               "RON": 10.0000 ,
                               "EADU": 2.00000 ,
                               "THRESH": 0.0500000 ,
                               "NZER": 8 ,
                               "DATADIR": "./"  ,
                               "STATIC": ""  ,
                               "RESULT": "donut.dat" ,
                               "PARFILE": "donut.par" ,
                               "IMFILE": "focus.fits" ,
                               "XC": 6869 ,
                               "YC": 4566 ,
                               "EFOC": 1
                            }
                        ]
                }

    def __init__(self):
        """
        Donut class constructor.
        """
        SExtractor.__init__(self)

        # Basic sextractor configuration for donut detection
        self.config['PIXEL_SCALE'] = 0.55
        self.config['BACK_TYPE'] = "AUTO"
        self.config['SATUR_LEVEL'] = 60000
        self.config['DETECT_MINAREA'] = 200
        self.config['DETECT_THRESH'] = 10.0
        self.config['VERBOSE_TYPE'] = "QUIET"
        self.config['PARAMETERS_LIST'] = ["NUMBER",'X_IMAGE', 'Y_IMAGE',
                                         "XWIN_IMAGE", "YWIN_IMAGE",
                                         "FLUX_BEST", "FWHM_IMAGE",
                                         "FLAGS"]

        self._useMPI = True
        self._mpithreads = 8
        self.setCFP(0.*units.meter)
        self.setPixScale(0.*units.mm)

        self.sign_x = +1.
        self.sign_y = +1.
        self.sign_u = +1.
        self.sign_v = +1.


    @units.quantity_input(cfp=units.meter)
    def setCFP(self,cfp):
        self._cfp = cfp

    @units.quantity_input(pix2mm=units.meter)
    def setPixScale(self,pix):
        self._pix = pix

    def getCFP(self):
        return self._cfp

    def getPixScale(self):
        return self._pix

    cfp = property(getCFP,setCFP)
    pixscale = property(getPixScale,setPixScale)

    def useMPI(self,opt=None):
        if opt is None:
            return self._useMPI
        elif type(opt) == bool:
            self._useMPI = opt
        else:
            log.warning('Input should be boolean or None. Got %s'%type(opt))
            return self._useMPI

    def setMPIThreads(self,n):
        self._mpithreads = n

    def getMPIThreads(self):
        return self._mpithreads

    def setupDonut(self,path=None):
        """
        Look for Donut program ('donut').
        If a full path is provided, only this path is checked.
        Raise a DonutException if it failed.
        Return program and version if it succeed.
        """

        candidates = ['donut','donut.py']

        if (path):
            candidates = [path]

        selected = None
        for candidate in candidates:
            try:
                cmd = '%s -h'%candidate
                p = Popen(candidate, shell=True,
                                     stdin=PIPE, stdout=PIPE,
                                     stderr=STDOUT, close_fds=True)
                (_out_err, _in) = (p.stdout, p.stdin)
                versionline = _out_err.read()
                if (versionline.find("Donut") != -1):
                    selected = candidate
                    break
            except IOError:
                continue

        if selected is None:
            raise MkOpticsException, \
                """
                  Cannot find Donut program. Check your PATH,
                  or provide the Donut program path in the constructor.
                  """

        _program = selected

        # print versionline
        _version_match = re.search("[Vv]ersion ([0-9\.])+", versionline)
        if not _version_match:
            raise MkOpticsException, \
                "Cannot determine Donut version."

        _version = _version_match.group()[8:]
        if not _version:
            raise MkOpticsException, \
                "Cannot determine Donut version."

        # print "Use " + self.program + " [" + self.version + "]"

        return _program, _version


    def align(self,ffile):
        '''

        :param file: Image to be processed.
        :return:
        '''

        log.debug('Running sextractor')
        self.run(ffile, clean=False)

        # Now calculates Zernike coeficients.
        outname = ffile.replace('.fits','_zern.npy')

        tmpCat = NamedTemporaryFile(delete=False)
        with open(tmpCat.name,'w') as fp:
            json.dump(self.ZERNPAR,fp)

        #program,version = self.setupDonut(os.path.expanduser('~/Develop/donut/script/donut'))
        program,version = os.path.expanduser('~/Develop/donut/script/donut'),"0.0"

        cmd = 'python %s -i %s -p %s -c %s -o %s'%(program,
                                            ffile,
                                            tmpCat.name,
                                            self.config['CATALOG_NAME'],
                                            outname)

        if self.useMPI():
            cmd = 'mpirun -np %i %s'%(self.getMPIThreads(),cmd)

        log.debug('Running donut with %i cores'%self.getMPIThreads())
        log.debug(cmd)
        p = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE)
        p.wait()

        log.info('Mapping hexapod position')

        hdr = fits.getheader(ffile)

        zmap = ZernMap(cfp = self.getCFP(),
                       pix2mm = self.getPixScale(),
                       center = [9216/2,9232/2] ) #[0,0]) # [hdr['NAXIS1']/2,hdr['NAXIS2']/2])

        rcat = np.load(outname).T

        cat = np.array([])

        for col in rcat:
            cat = np.append(cat,np.array([col.reshape(-1),]))
        cat = cat.reshape(rcat.shape[0],-1)

        fitmask = cat[0] == 1
        cat = cat[1:]

        log.debug('cat.shape: %s'%cat.shape.__str__())

        id_seeing = 2
        id_focus = 5
        id_astigx = 6
        id_astigy = 7
        id_commay = 8
        id_commax = 9

        planeU = zmap.astigmatism(cat[0],cat[1],cat[id_astigx]*self.config['PIXEL_SCALE'],0)
        planeV = zmap.astigmatism(cat[0],cat[1],cat[id_astigy]*self.config['PIXEL_SCALE'],1)
        log.debug('cat[0]/cat[%i]: %s'%(id_commax,cat[id_commay].shape.__str__()))
        comaX,mask = zmap.comma(cat[0],cat[id_commax]*self.config['PIXEL_SCALE'])
        comaY,mask = zmap.comma(cat[1],cat[id_commay]*self.config['PIXEL_SCALE'])
        focus = zmap.map(cat[0],cat[1],cat[id_focus]*self.config['PIXEL_SCALE'])
        seeing = zmap.map(cat[0],cat[1],cat[id_seeing])

        hOffSet = HexapodAxes()

        hOffSet.u = self.sign_u*(planeU['U']+planeV['U'])/2.
        hOffSet.v = self.sign_v*(planeU['V']+planeV['V'])/2.
        hOffSet.x = self.sign_x*np.poly1d(comaX)(0.)*units.mm
        hOffSet.y = self.sign_y*np.poly1d(comaY)(0.)*units.mm
        hOffSet.z = focus[2]/10.*units.mm
        hOffSet.seeing = seeing[2]*units.arcsec

        return hOffSet

