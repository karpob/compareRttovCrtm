#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import configparser, os, sys, h5py
import numpy as np
from matplotlib import pyplot as plt

pathInfo = configparser.ConfigParser()
# Stuff to get the installed rttov path, and import pyrttov interface
pathInfo.read('rttov.cfg')
rttovPath = pathInfo['RTTOV']['rttovPath']
pyRttovPath = os.path.join(rttovPath,'wrapper')
if not pyRttovPath in sys.path:
    sys.path.append(pyRttovPath)
import pyrttov
# pull in example from basic rttov example
import example_data as ex
# import pycrtm f2py module
from lib.pycrtm.pyCRTM import pyCRTM, profilesCreate
from lib.pycrtm.crtm_io import readTauCoeffODPS

def interpolateProfile(x, xo, yo):
    """
    Do a log-linear interpolation.
    """
    logX =  np.log(x)
    logXo = np.log(xo)
    logYo = np.log(yo)
    return np.exp(np.interp(logX, logXo, logYo))

 
def crtmLevelsToLayers( pLevels ):
    num = pLevels[1::] - pLevels[0:pLevels.shape[0]-1]
    den = np.log(pLevels[1::]/pLevels[0:pLevels.shape[0]-1])
    return num/den


# helper function from pyrttov example 
def expand2nprofiles(n, nprof):
    # Transform 1D array to a [nprof, nlevels] array
    outp = np.empty((nprof, len(n)), dtype=n.dtype)
    for i in range(nprof):
            outp[i, :] = n[:]
    return outp
def readProfileH5(filename,nprofiles,coefLevs):
    h5 = h5py.File( filename )
    groups = []
    #nlevs = np.asarray(coefLevs).shape[0]
    nlevs = 101
    for i in range(1,nprofiles+1):
        groups.append("{:04d}".format(i))
    P = np.zeros([nprofiles,101])
    T = np.zeros([nprofiles,101])
    Q = np.zeros([nprofiles,101])
    CO2 = np.zeros([nprofiles,101])
    O3 = np.zeros([nprofiles,101])

    Po = np.zeros([nprofiles,nlevs])
    To = np.zeros([nprofiles,nlevs])
    Qo = np.zeros([nprofiles,nlevs])
    CO2o = np.zeros([nprofiles,nlevs])
    O3o = np.zeros([nprofiles,nlevs])
    print(list(h5['PROFILES'].keys()))
    groups = ['0003','0003']
    for i,g in enumerate(groups):
        print ('ggg', i, g)
        P[i,:] = np.asarray(h5['PROFILES'][g]['P'])
        Q[i,:] = np.asarray(h5['PROFILES'][g]['Q'])
        Po[i,:] = P[i,:] #coefLevs
        T[i,:] = np.asarray(h5['PROFILES'][g]['T'])
        To[i,:] = T[i,:]#interpolateProfile(Po[i,:], P[i,:], T[i,:])
        Qo[i,:] = Q[i,:]#interpolateProfile(Po[i,:], P[i,:], Q[i,:])
        CO2[i,:] = np.asarray(h5['PROFILES'][g]['CO2'])
        CO2o[i,:] = CO2[i,:] #interpolateProfile(Po[i,:], P[i,:], CO2[i,:])
        O3[i,:] = np.asarray(h5['PROFILES'][g]['O3'])
        O3o[i,:] = O3[i,:] #interpolateProfile(Po[i,:], P[i,:], O3[i,:])
        GasUnits = int(np.asarray(h5['PROFILES'][g]['GAS_UNITS']))
    
    return Po, To, Qo, CO2o, O3o, GasUnits 



if __name__ == "__main__":
    
    coefLev54 = [0.0050, 0.0131, 0.0304, 0.0644, 0.1263, 0.2324, 0.4052, 0.6749, 1.0801, 1.6691,2.5011, 3.6462, 5.1864, 7.2150, 9.8368,\
               13.1672, 17.3308, 22.4601, 28.6937, 36.1735, 45.0430, 55.4433, 67.5109, 81.3744, 97.1505, 114.9420, 134.8320, 156.8850, 181.1390,\
               207.6090, 236.2780, 267.1010, 300.0000, 334.8650, 371.5530, 409.8890, 449.6680, 490.6520, 532.5770, 575.1540, 618.0710, 660.9960,\
               703.5860, 745.4840, 786.3280, 825.7550, 863.4050, 898.9280, 931.9850, 962.2590, 989.4510, 1013.2900, 1033.5400, 1050.0000]
    
    #coefLev = np.asarray(h5['boundary_pressures'])
    #layerPressures = np.asarray(h5['layer_pressures'])
    # get installed path to coefficients from pycrtm submodule install (crtm.cfg in pycrtm directory)
    pathToThisScript = os.path.dirname(os.path.abspath(__file__))
    pathInfo = configparser.ConfigParser()
    pathInfo.read( os.path.join(pathToThisScript,'lib','pycrtm','crtm.cfg') )
    coefficientPathCrtm = pathInfo['CRTM']['coeffs_dir']
    # get pressures used for profile training in CRTM.
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'cris-fsr_n20.TauCoeff.bin') ) 

    coefLev = np.asarray(crtmTauCoef['level_pressure'])
    layerPressures = np.asarray(crtmTauCoef['layer_pressure'])
    crtmTauCoef = []
    print ('layer pressure',layerPressures) 
    print ('coefLev',coefLev) 
    #### Profile Loading junk from the rttov example ####
    
    # Declare an instance of Profiles
    nlevels = 101  
    nprofiles = 2
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles2 = pyrttov.Profiles(nprofiles, nlevels-3)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas_kgkg.H5'
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFilename, nprofiles, layerPressures)
    myProfiles2.P = myProfiles.P[:,0:98]
    myProfiles2.Q = myProfiles.Q[:,0:98]
    myProfiles2.T = myProfiles.T[:,0:98]
    myProfiles2.CO2 = myProfiles.CO2[:,0:98]
    myProfiles2.O3 =  myProfiles.O3[:,0:98]
    myProfiles2.GasUnits =  myProfiles.GasUnits
    myProfiles = pyrttov.Profiles(nprofiles, nlevels-3)
    myProfiles = myProfiles2
    # Associate the profiles and other data from example_data.h with myProfiles
    # Note that the simplecloud, clwscheme, icecloud and zeeman data are not mandatory and
    # are omitted here
    # 
    # satzen, satazi, sunzen, sunazi
    myProfiles.Angles = 0.0*ex.angles.transpose()
    myProfiles.Angles[:,2] = 100.0 #below horizon for solar
    myProfiles.S2m = np.array([[myProfiles.P[0,-1], myProfiles.T[0,-1] ,myProfiles.Q[0,-1] , 0, 0., 0.],\
                [myProfiles.P[0,-1], myProfiles.T[0,-1] ,myProfiles.Q[1,-1] , 0., 0., 0.]], dtype=np.float64)
    myProfiles.Skin = 0.0*ex.skin.transpose()
    myProfiles.Skin[:,0] = 270.0 #myProfiles.T[0,-1]
    myProfiles.Skin[:,1] = 35.0
    # make SurfType to Water surface over ocean
    myProfiles.SurfType = np.ones(ex.surftype.shape).transpose()
    myProfiles.SurfGeom = np.zeros(ex.surfgeom.shape).transpose()
    myProfiles.DateTimes = ex.datetimes.transpose()
    #### End Profile loading junk from rttov example #####
    # ------------------------------------------------------------------------
    # Set up Rttov instances for each instrument
    # ------------------------------------------------------------------------

    # Create Rttov object
    rttovObj = pyrttov.Rttov()
    rttovObj.Options.AddInterp = True
    rttovObj.Options.StoreTrans = True
    rttovObj.Options.StoreRad = True
    rttovObj.Options.StoreEmisTerms = True
    rttovObj.Options.VerboseWrapper = True
    rttovObj.Options.CO2Data = True
    rttovObj.Options.OzoneData = True
    print( rttovObj.Options.IrSeaEmisModel)
    rttovObj.Options.IrSeaEmisModel =2
    #rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov8pred101L','rtcoef_jpss_0_cris-fsr.H5')
    rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov8pred101L','rtcoef_metop_2_iasi.H5')
    # set Fastem version to 5 for comparison with CRTM.
    rttovObj.Options.FastemVersion = int(5)
    #load the coefficient
    rttovObj.loadInst()
    
    #load the profiles
    rttovObj.Profiles = myProfiles
    # have rttov calculate surface emission.
    h5 = h5py.File('whir.h5','r')
    print('nchannels', rttovObj.Nchannels)
    rttovObj.SurfEmisRefl = -1.0*np.ones((2,nprofiles,8461), dtype=np.float64)
    #rttovObj.SurfEmisRefl = h5['emis']*np.ones((2,nprofiles,8461), dtype=np.float64)
    
    # run it 
    rttovObj.runDirect()
    print("surf emis",rttovObj.SurfEmisRefl)
    print(rttovObj.Bt)

    print("Now on to CRTM.")
    """
    Do this in a loop. Kind of a hack, but can consolidate into 
    openmp call in fortran at some point if we deal with lots of profiles.
    """
    # Associate the profiles and other data from example_data.h with myProfiles
    # Note that the simplecloud, clwscheme, icecloud and zeeman data are not mandatory and
    # are omitted here
    # 
    # satzen, satazi, sunzen, sunazi
    nprofiles = 2
    
    myProfiles = pyrttov.Profiles(nprofiles, 101)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas_kgkg.H5'
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFilename, nprofiles, layerPressures)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas.H5'
    _, _, _, CO2_1, O3_1, units_1 = readProfileH5(h5ProfileFilename, nprofiles, layerPressures)
    print(CO2_1, O3_1,'units!',units_1)
    profilesCRTM = profilesCreate( 2, 98 )
    profilesCRTM.Angles[:,0] = 0.0
    profilesCRTM.Angles[:,1] = 0.0 
    profilesCRTM.Angles[:,2] = 100.0  # 100 degrees zenith below horizon.
    profilesCRTM.Angles[:,3] = 0.0 # zero solar azimuth 
    profilesCRTM.Angles[:,4] = 0.0 
    dt = ex.datetimes.transpose()
    print('print p levels', myProfiles.P[0,0:98]) 
    profilesCRTM.DateTimes[:,0:3] = dt[:,0:3]
    profilesCRTM.Pi[:,:] = myProfiles.P[:,0:99]
    profilesCRTM.P[:,:] = myProfiles.P[:,0:98]
    profilesCRTM.T[:,:] = myProfiles.T[:,0:98]
    profilesCRTM.Q[:,:] = 1000.0*myProfiles.Q[:,0:98]
    profilesCRTM.O3[:,:] = O3_1[:,0:98]
    profilesCRTM.CO2[:,:] = CO2_1[:,0:98]
    profilesCRTM.aerosolType[:] = -1
    profilesCRTM.cloudType[:] = -1
    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 270 # profilesCRTM.T[0,-1] #h5['surfaceTemperatures']
    profilesCRTM.S2m[:,1] = 35.0 # just use salinity out of S2m for the moment.
    profilesCRTM.windSpeed10m[:] = 0.0
    profilesCRTM.windDirection10m[:] = 0.0 
    profilesCRTM.n_absorbers[:] = 2 #h5['n_absorbers'][()]
    # land, soil, veg, water, snow, ice
    #profilesCRTM.surfaceTypes[i,0] = #h5['landType'][()]

    #profilesCRTM.surfaceTypes[i,1] = #h5['soilType'][()]
    #profilesCRTM.surfaceTypes[i,2] = #h5['vegType'][()]
    profilesCRTM.surfaceTypes[:,3] = 1 #h5['waterType'][()]
    #profilesCRTM.surfaceTypes[i,4] = #h5['snowType'][()]
    #profilesCRTM.surfaceTypes[i,5] = #h5['iceType'][()]

    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    #crtmOb.sensor_id = 'cris-fsr_n20'
    crtmOb.sensor_id = 'iasi_metop-b' 
    crtmOb.nThreads = 2

    crtmOb.loadInst()

    crtmOb.runDirect()
    h5 = h5py.File('iasi_wavenumbers.h5')
    idx =np.asarray( h5['idxBufrSubset'])-1 
    for i in range(8461):
        print(i+1, crtmOb.Bt[0,i],rttovObj.Bt[0,i],  crtmOb.Bt[0,i]-rttovObj.Bt[0,i])
    print(dir(crtmOb))
    for i in range(8461):

        print(i+1, crtmOb.surfEmisRefl.shape ,rttovObj.SurfEmisRefl.shape)
        print(i+1, crtmOb.surfEmisRefl[0,i],rttovObj.SurfEmisRefl[0,0,i],  crtmOb.surfEmisRefl[0,i]-rttovObj.SurfEmisRefl[0,0,i])
    print(idx) 
    wv = np.asarray(crtmOb.Wavenumbers)[idx]
    plt.figure()
    plt.plot(wv, crtmOb.Bt[0,idx],'b')
    plt.plot(wv, rttovObj.Bt[0,idx],'r')
    plt.savefig('whir.png')


    plt.figure()
    plt.plot(wv, crtmOb.Bt[0,idx]-rttovObj.Bt[0,idx],'k')
    plt.savefig('whirDelta.png')


    plt.figure()
    plt.plot(wv, crtmOb.surfEmisRefl[0,idx],'b')
    plt.plot(wv, rttovObj.SurfEmisRefl[0,0,idx],'r')
    plt.savefig('whir2.png')

    plt.figure()
    plt.plot(wv, crtmOb.surfEmisRefl[0,idx]- rttovObj.SurfEmisRefl[0,0,idx],'k')
    plt.savefig('whir3.png')
    #h5out = h5py.File('whir.h5','w')
    #h5out.create_dataset('emis',data=crtmOb.surfEmisRefl[0,:])
    #h5out.close()
