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
    
    # get installed path to coefficients from pycrtm submodule install (crtm.cfg in pycrtm directory)
    # load stuff we need for CRTM coefficients
    pathToThisScript = os.path.dirname(os.path.abspath(__file__))
    pathInfo = configparser.ConfigParser()
    pathInfo.read( os.path.join(pathToThisScript,'lib','pycrtm','crtm.cfg') )
    coefficientPathCrtm = pathInfo['CRTM']['coeffs_dir']
    
    # Trying to make the interpolation here instead of inside CRTM, pull pressure levels out of coefficient
    # get pressures used for profile training in CRTM.
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'iasi_metop-b.TauCoeff.bin') ) 
    coefLevCrtm = np.asarray(crtmTauCoef['level_pressure'])
    layerPressuresCrtm = np.asarray(crtmTauCoef['layer_pressure'])
    crtmTauCoef = [] # clear out some ram. by getting rid of the dictonary and set it to empty list
    
    ##########################
    # Start Profile Setting
    ##########################
    # Declare an instance of Profiles
    nlevels = 101  
    nprofiles = 2
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles2 = pyrttov.Profiles(nprofiles, nlevels-3)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas_kgkg.H5'
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFilename, nprofiles, layerPressuresCrtm)
    myProfiles2.P = myProfiles.P[:,0:98]
    myProfiles2.Q = myProfiles.Q[:,0:98]
    myProfiles2.T = myProfiles.T[:,0:98]
    myProfiles2.CO2 = myProfiles.CO2[:,0:98]
    myProfiles2.O3 =  myProfiles.O3[:,0:98]
    myProfiles2.GasUnits =  myProfiles.GasUnits
    myProfiles = pyrttov.Profiles(nprofiles, nlevels-3)
    myProfiles = myProfiles2

    # View/Solar angles
    # satzen, satazi, sunzen, sunazi
    #nprofiles, nvar (4)
    myProfiles.Angles = 0.0*np.zeros([nprofiles,4])
    # set solar zenith angle below horizon +10 deg
    myProfiles.Angles[:,2] = 100.0 #below horizon for solar

    # s2m surface 2m variables
    # surface : pressure, temperature, specific humidity, wind (u comp), wind (v comp), windfetch 
    # nprofiles, nvar (6)
    s2m = []
    for i in list(range(nprofiles)):
        s2m.append([myProfiles.P[i,-1], myProfiles.T[i,-1] ,myProfiles.Q[i,-1] , 0, 0., 0.])
    myProfiles.S2m = np.asarray(s2m)
    
    #Skin variables
    #skin%t, skin%salinity, skin%snow_fraction, skin %foam_fraction, skin%fastem(1:5), skin%specularity
    #nprofiles, nvar (10)
    myProfiles.Skin = np.zeros([nprofiles,10])
    myProfiles.Skin[:,0] = 270.0
    myProfiles.Skin[:,1] = 35.0

    #Surface Type info
    # surftype (land = 0, sea = 1, seacice =2), watertype (fresh = 0, ocean = 1)
    #nprofiles, nvar (2)
    myProfiles.SurfType = np.ones([nprofiles,2])

    #Surface geometry
    # latitude, longitude, elevation (lat/lon used in refractivity and emis/BRDF atlases, elevation used for refractivity).
    #nprofiles, nvar (3)
    myProfiles.SurfGeom = np.zeros([nprofiles,3])

    #Date/times 
    #nprofiles, nvar(6)
    datetimes = []
    for i in list(range(nprofiles)):
        datetimes.append( [2015 ,   8,    1,    0,    0,    0]) 
    myProfiles.DateTimes = np.asarray(datetimes)

    ##########################
    # End profile setting
    ##########################

    #########################
    # Run RTTOV
    #########################

    # Create Rttov object
    rttovObj = pyrttov.Rttov()
    # set options.
    rttovObj.Options.AddInterp = True
    rttovObj.Options.StoreTrans = True
    rttovObj.Options.StoreRad = True
    rttovObj.Options.StoreEmisTerms = True
    rttovObj.Options.VerboseWrapper = True
    rttovObj.Options.CO2Data = True
    rttovObj.Options.OzoneData = True
    rttovObj.Options.IrSeaEmisModel =2
    rttovObj.Options.FastemVersion = int(6)
    rttovObj.Options.Nthreads = 2
    rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov8pred101L','rtcoef_metop_2_iasi.H5')

    #load the coefficient
    rttovObj.loadInst()

    #load the profiles
    rttovObj.Profiles = myProfiles

    # have rttov calculate surface emission.
    rttovObj.SurfEmisRefl = -1.0*np.ones((2,nprofiles,rttovObj.Nchannels), dtype=np.float64)
    
    # run it 
    rttovObj.runDirect()
    
    ########################
    # End Run RTTOV
    ########################

    print("Now on to CRTM.")
    
    ##############################
    # Begin CRTM Profile setting
    ##############################
    nprofiles = 2
    myProfiles = pyrttov.Profiles(nprofiles, 101)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas_kgkg.H5'
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFilename, nprofiles, layerPressuresCrtm)
    h5ProfileFilename =  '/Users/bkarpowi/github/compareRttovCrtm/rttovDir/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas.H5'
    _, _, _, CO2_1, O3_1, units_1 = readProfileH5(h5ProfileFilename, nprofiles, layerPressuresCrtm)
    profilesCRTM = profilesCreate( 2, 98 )
    profilesCRTM.Angles[:,:] = 0.0
    profilesCRTM.Angles[:,2] = 100.0  # Solar Zenith Angle 100 degrees zenith below horizon.

    profilesCRTM.DateTimes[:,0] = 2015
    profilesCRTM.DateTimes[:,1] = 8
    profilesCRTM.DateTimes[:,2] = 1

    profilesCRTM.Pi[:,:] = myProfiles.P[:,0:99]
    profilesCRTM.P[:,:] = myProfiles.P[:,0:98]
    profilesCRTM.T[:,:] = myProfiles.T[:,0:98]
    profilesCRTM.Q[:,:] = 1000.0*myProfiles.Q[:,0:98]
    profilesCRTM.O3[:,:] = O3_1[:,0:98]
    profilesCRTM.CO2[:,:] = CO2_1[:,0:98]

    # Turn off Aerosols and Clouds
    profilesCRTM.aerosolType[:] = -1
    profilesCRTM.cloudType[:] = -1

    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 270 
    profilesCRTM.S2m[:,1] = 35.0 # just use salinity out of S2m for the moment.
    profilesCRTM.windSpeed10m[:] = 0.0
    profilesCRTM.windDirection10m[:] = 0.0 
    profilesCRTM.n_absorbers[:] = 2 

    # land, soil, veg, water, snow, ice
    profilesCRTM.surfaceTypes[:,3] = 1 

    ##############################
    # End CRTM Profile setting
    ##############################

    #########################
    # Run CRTM
    #########################
    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    crtmOb.sensor_id = 'iasi_metop-b' 
    crtmOb.nThreads = 2

    crtmOb.loadInst()
    crtmOb.runDirect()

    #########################
    # End Run CRTM
    #########################


    # RT is done! Great! Let's make some plots!

    # get the 616 channel subset for IASI
    h5 = h5py.File('iasi_wavenumbers.h5')
    idx =np.asarray( h5['idxBufrSubset'])-1 
    wv = np.asarray(crtmOb.Wavenumbers)[idx]
    
    ##############################
    # Start Plots
    ##############################
    plt.figure()
    plt.plot(wv, crtmOb.Bt[0,idx],'b',label='CRTM')
    plt.plot(wv, rttovObj.Bt[0,idx],'r',label='RTTOV')
    plt.xlabel('Wavenumber [cm$^{-1}$]')
    plt.ylabel('Brightness Temperature [K]')
    plt.legend()
    plt.savefig('iasi_crtm_rttov.png')

    plt.figure()
    plt.plot(wv, crtmOb.Bt[0,idx]-rttovObj.Bt[0,idx],'k')
    plt.xlabel('Wavenumber [cm$^{-1}$]')
    plt.ylabel('CRTM - RTTOV Brightness Temperature [K]')
    plt.savefig('iasi_crtm_rttov_diff.png')

    plt.figure()
    plt.plot(wv, crtmOb.surfEmisRefl[0,idx],'b', label='CRTM')
    plt.plot(wv, rttovObj.SurfEmisRefl[0,0,idx],'r', label='RTTOV')
    plt.legend()
    plt.xlabel('Wavenumber [cm$^{-1}$]')
    plt.ylabel('Emissivity')
    plt.savefig('iasi_emissivity_crtm_rttov.png')

    plt.figure()
    plt.plot(wv, crtmOb.surfEmisRefl[0,idx]- rttovObj.SurfEmisRefl[0,0,idx],'k')
    plt.xlabel('Wavenumber [cm$^{-1}$]')
    plt.ylabel('CRTM - RTTOV Emissivity')
    plt.savefig('iasi_emissivity_crtm_rttov_diff.png')
    ##############################
    # End Plots
    ##############################
