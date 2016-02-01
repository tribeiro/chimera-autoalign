
import numpy as np
import ccdproc
from ccdproc.utils.slices import slice_from_string

import ConfigParser

import logging

log = logging.getLogger(__name__)

class CCDSection(object):

    def __init__(self):
        self.serial_index = 0
        self.parallel_index = 0

        self.serial_size = 0
        self.parallel_size= 0

        self.sec = None

        self.serial_scans = []
        self.serial_scans_correct = []

        self.parallel_scans = []
        self.parallel_scans_correct = []

    def get_section(self):
        return self.sec

    def set_section(self, value):
        if type(value) != str:
            raise TypeError('Input type must be string. Got %s' % type(value))

        self.sec = slice_from_string(value)
        self.serial_size = self.sec[1].stop - self.sec[1].start
        self.parallel_size = self.sec[0].stop - self.sec[0].start

    section = property(get_section, set_section)

    def configOverScanRegions(self):

        def parse_scan(scan):
            # print scan
            # Try to parse it to integer
            try:
                reg_len = int(scan[1])
                if scan[0] == 'serial_pre':
                    log.debug('Serial pre-scan with %i length.' % reg_len)
                    return self.section[0], slice(self.section[1].start-reg_len,self.section[1].stop)
                elif scan[0] == 'serial_pos':
                    log.debug('Serial pos-scan with %i length.' % reg_len)
                    return self.section[0], slice(self.section[1].stop,self.section[1].stop+reg_len)
                elif scan[0] == 'parallel_pre':
                    log.debug('Parallel pre-scan with %i length.' % reg_len)
                    return slice(self.section[0].start-reg_len,self.section[0].start), self.section[1]
                elif scan[0] == 'parallel_pos':
                    log.debug('Parallel pos-scan with %i length.' % reg_len)
                    return slice(self.section[0].stop,self.section[0].stop+reg_len), self.section[1]
                else:
                    raise KeyError('%s is not a valid key. Should be one of serial_pre, serial_pos, '
                                   'parallel_pre or parallel_pos' % (scan[0]))
            except ValueError, e:
                log.debug('Region is not convertible to int. Falling back to slice mode.')
                return slice_from_string(scan[1])

        for i_serial in range(len(self.serial_scans)):
            self.serial_scans[i_serial] = parse_scan(self.serial_scans[i_serial])

        for i_parallel in range(len(self.parallel_scans)):
            self.parallel_scans[i_parallel] = parse_scan(self.parallel_scans[i_parallel])

    def __str__(self):
        return 'CCDSection: %s has %i serial and %i parallel overscans' % (self.section,
                                                                           len(self.serial_scans),
                                                                           len(self.parallel_scans))

class OverscanCorr():

    def __init__(self,*args,**kwargs):

        self.ccd = None

        self._serialports = 1
        self._parallelports = 1

        self._ccdsections = []

        self._serial_prescan_correct = []
        self._serial_posscan_correct = []
        self._parallel_prescan_correct = []
        self._parallel_posscan_correct = []

        if 'config' in kwargs.keys():
            self.loadConfiguration(kwargs['config'])

    def read(self,filename):
        self.ccd = ccdproc.CCDData.read(filename,unit='electron')

    def loadConfiguration(self,configfile):
        '''
        Sample configuration file.

        [ccdconfig]
        serialports: 1
        parallelports: 1

        [section_1_1]
        region: [6:506,6:506]
        serialprescan: 6        # if int will consider overscan as comming for 1st pixel
        serialprescancorr: True
        serialposscan: [506:512,6:506] # define as a image slice
        serialposcancorr: True
        parallelprescan: [6:506,0:6] # Again with image slice
        parallelprescancorr: True
        parallelposscan: 6 # again with number of pixels, but now postscan. Start at the last pixel in the science region
        parallelposcancorr: True

        :param configfile:
        :return:
        '''

        config = ConfigParser.RawConfigParser()

        config.read(configfile)

        if config.has_section('ccdconfig'):
            self._serialports = int(config.get('ccdconfig', 'serialports'))
            self._parallelports = int(config.get('ccdconfig', 'parallelports'))

        for serial in range(self._serialports):
            for parallel in range(self._parallelports):

                ccdsec = CCDSection()
                ccdsec.serial_index = serial
                ccdsec.parallel_index = parallel
                ccdsec.section = config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                    'region')
                ccdsec.serial_scans.append(('serial_pre', config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                          'serialprescan')))
                ccdsec.serial_scans_correct.append(bool(config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                                    'serialprescancorr')))
                ccdsec.serial_scans.append(('serial_pos', config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                          'serialposscan')))
                ccdsec.serial_scans_correct.append(bool(config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                                    'serialposscancorr')))
                ccdsec.parallel_scans.append(('parallel_pre', config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                          'parallelprescan')))
                ccdsec.parallel_scans_correct.append(bool(config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                                    'parallelprescancorr')))
                ccdsec.parallel_scans.append(('parallel_pos', config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                          'parallelposscan')))
                ccdsec.parallel_scans_correct.append(bool(config.get('section_%i_%i' % ( serial+1, parallel+1),
                                                                    'parallelposscancorr')))
                # print ccdsec.section, ccdsec.parallel_size, ccdsec.serial_size
                ccdsec.configOverScanRegions()

                self._ccdsections.append(ccdsec)

        # self.configOverScanRegions()

    # Todo: Check that ccdsections actually makes sense.
    # Do they follow the serial/parallel order?
    # Do they overlap with each other?


    # def configOverScanRegions(self):
    #     '''
    #     Parse overscan regions to build image slices. If they where gives as slices, test they are ok.
    #     :return:
    #     '''
    #
    #     for i_scan in range(len(self.scans)):
    #         scan = self.scans[i_scan]
    #         for i_region in range(len(scan[1])):
    #             # Try to parse it to integer
    #             region = None
    #             try:
    #                 reg_len = int(scan[1][i_region])
    #                 if scan[0] == 'serial_pre':
    #                     log.debug('Serial pre-scan with %i length.' % reg_len)
    #                     region = (slice(None,None),slice(None,reg_len))
    #                 elif scan[0] == 'serial_pos':
    #                     log.debug('Serial pos-scan with %i length.' % reg_len)
    #                     region = (slice(None,None),slice(-reg_len,None))
    #                 elif scan[0] == 'parallel_pre':
    #                     log.debug('Parallel pre-scan with %i length.' % reg_len)
    #                     region = (slice(None,reg_len),slice(None,None))
    #                 elif scan[0] == 'parallel_pos':
    #                     log.debug('Parallel pos-scan with %i length.' % reg_len)
    #                     region = (slice(-reg_len,None),slice(None,None))
    #                 else:
    #                     raise KeyError('%s is not a valid key. Should be one of serial_pre, serial_pos, '
    #                                    'parallel_pre or parallel_pos' % (scan[0]))
    #             except ValueError, e:
    #                 log.debug('Region is not convertible to int. Falling back to slice mode.')
    #                 region = slice_from_string(region)
    #                 pass
    #             self.scans[i_scan][1][i_region] = region
    #
    #     # Todo: Check that overscan region actually makes sense, i.e. do not overlaps with science array.

    def show(self):
        '''
        Display image of the CCD data with overplotted regions.

        :return:
        '''
        import pyds9 as ds9

        d = ds9.ds9()

        # print self._ccdsections[0].parallel_scans[1]

        d.set_np2arr(self.ccd.data[self._ccdsections[0].parallel_scans[1]])

    def overscan(self):
        '''
        Subtract overscan from current CCDData image.

        :return: Nothing.
        '''

        pass

    def trim(self):
        '''
        Return trimmed CCDData object.

        :return: CCDData object with trimmed array.
        '''

        log.debug('CCD has %i subarrays in a %i x %i matrix.' % (self._parallelports*self._serialports,
                                                                 self._parallelports,self._serialports))

        subarraysize = [0,0]
        for subarr in self._ccdsections:
            subarraysize[0] = max(subarraysize[0],subarr.parallel_size)
            subarraysize[1] = max(subarraysize[1],subarr.serial_size)
        log.debug('Trimmed CCD will be %i x %i' % (subarraysize[0]*self._parallelports,
                                                   subarraysize[1]*self._serialports))
        newdata = np.zeros((subarraysize[0]*self._parallelports,
                                                   subarraysize[1]*self._serialports))

        for subarr in self._ccdsections:
            newdata[subarraysize[0]*subarr.parallel_index:
                    subarraysize[0]*(subarr.parallel_index+1),
                    subarraysize[1]*subarr.serial_index:
                    subarraysize[1]*(subarr.serial_index+1)] += self.ccd.data[subarr.section]
        newccd = self.ccd
        newccd.data = newdata

        return newccd

if __name__ == '__main__':

    import sys
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option('-f','--filename',
                      help = 'Input image name.',
                      type='string')

    parser.add_option('-c','--config',
                      help = 'Configuration file.',
                      type='string')

    parser.add_option('-o','--output',
                      help = 'Output name.',
                      type='string')

    opt, args = parser.parse_args(sys.argv)

    logging.basicConfig(format='%(levelname)s:%(asctime)s::%(message)s',
                        level=logging.DEBUG)

    logging.info('Reading in %s' % opt.filename)

    overcorr = OverscanCorr()

    overcorr.read('%s' % opt.filename)

    logging.info('Loading configuration from %s' % opt.config)

    overcorr.loadConfiguration(opt.config)

    overcorr.show()

    logging.info('Applying overscan...')

    overcorr.overscan()

    logging.info('Trimming...')

    ccdout = overcorr.trim()

    if opt.output is not None:
        logging.info('Saving result to %s...' % opt.output)

        ccdout.write(opt.output)

