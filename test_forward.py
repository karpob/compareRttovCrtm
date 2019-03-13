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
from lib.pycrtm import pycrtm
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
    nlevs = np.asarray(coefLevs).shape[0]
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

    for i,g in enumerate(groups):
        P[i,:] = np.asarray(h5['PROFILES'][g]['P'])
        Q[i,:] = np.asarray(h5['PROFILES'][g]['Q'])
        Po[i,:] = coefLevs
        T[i,:] = np.asarray(h5['PROFILES'][g]['T'])
        To[i,:] = interpolateProfile(Po[i,:], P[i,:], T[i,:])
        Qo[i,:] = interpolateProfile(Po[i,:], P[i,:], Q[i,:])
        CO2[i,:] = np.asarray(h5['PROFILES'][g]['CO2'])
        CO2o[i,:] = interpolateProfile(Po[i,:], P[i,:], CO2[i,:])
        O3[i,:] = np.asarray(h5['PROFILES'][g]['O3'])
        O3o[i,:] = 0.0*interpolateProfile(Po[i,:], P[i,:], O3[i,:])
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
 
    #### Profile Loading junk from the rttov example ####
    
    # Declare an instance of Profiles
    nlevels = 100  
    nprofiles = 2
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles2 = pyrttov.Profiles(nprofiles, nlevels-1)
    h5ProfileFilename =  '/discover/nobackup/bkarpowi/rt3/rttov/rttov_test/profile-datasets-hdf/standard101lev_allgas_kgkg.H5'
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFilename, nprofiles, layerPressures)
    # Associate the profiles and other data from example_data.h with myProfiles
    # Note that the simplecloud, clwscheme, icecloud and zeeman data are not mandatory and
    # are omitted here
    # 
    # satzen, satazi, sunzen, sunazi
    myProfiles.Angles = 0.0*ex.angles.transpose()
    myProfiles.Angles[:,0] = 0
    myProfiles.Angles[:,1] = 0
    myProfiles.S2m = np.array([[myProfiles.P[0,-1],270.0 ,myProfiles.Q[0,-1] , 0, 0., 0.],\
                [myProfiles.P[0,-1],270.0 ,myProfiles.Q[1,-1] , 0., 0., 0.]], dtype=np.float64)
    myProfiles.Skin = ex.skin.transpose()
    # make SurfType to Water surface over ocean
    myProfiles.SurfType = np.ones(ex.surftype.shape).transpose()
    myProfiles.SurfGeom = ex.surfgeom.transpose()
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
    rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov9pred101L','rtcoef_jpss_0_cris-fsr.H5')
    # set Fastem version to 5 for comparison with CRTM.
    rttovObj.Options.FastemVersion = int(5)
    #load the coefficient
    rttovObj.loadInst()
    
    #load the profiles
    rttovObj.Profiles = myProfiles
    # have rttov calculate surface emission. 
    rttovObj.SurfEmisRefl =-1.0*np.ones((2,nprofiles,2211), dtype=np.float64)
    
    # run it 
    rttovObj.runDirect()
    print("surf emis",rttovObj.SurfEmisRefl)
    print(rttovObj.Bt)

    print("Now on to CRTM.")
    """
    Do this in a loop. Kind of a hack, but can consolidate into 
    openmp call in fortran at some point if we deal with lots of profiles.
    """
    crtmTb = [] 
    # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
    for iprof in range(0,2):
        print(iprof)
        zenithAngle = np.asarray(myProfiles.Angles[iprof][0])
        azimuthAngle = 0.0#np.asarray(myProfiles.Angles[iprof][1])
        scanAngle = 0.0 #np.asarray(zenithAngle)
        solarAngle =np.asarray( myProfiles.Angles[iprof][2])
        print( zenithAngle, azimuthAngle, scanAngle, solarAngle)
        nChan = int(2211)
        pressureLevels = np.asarray(coefLev) #np.asarray(myProfiles.P[iprof,0::])
        pressureLayers = layerPressures
        temperatureLayers = myProfiles.T[iprof,:]#interpolateProfile(pressureLayers, pressureLevels, myProfiles.T[iprof,:])
        #_, _, myProfiles.Q, _, _, _ = readProfileH5(h5ProfileFilename, nprofiles, coefLev)

        humidityLayers = 1000.0*myProfiles.Q[iprof,:]#interpolateProfile(pressureLayers, pressureLevels, myProfiles.Q[iprof,:])
        ozoneConcLayers = 0.0*myProfiles.O3[iprof,:]#interpolateProfile(pressureLayers, pressureLevels, myProfiles.O3[iprof,:]) 
        surfaceType = 0 
        surfaceTemperature = 270.0
        windSpeed10m = 0.0#np.sqrt(4.**2 + 2.**2)
        windDirection10m = 0.0#np.arctan2(4./windSpeed10m,2/windSpeed10m)*(180.0/np.pi)
        Tb, Transmission = pycrtm.pycrtm.wrap_forward( coefficientPathCrtm, 'cris-fsr_n20',\
                        zenithAngle, scanAngle, azimuthAngle, solarAngle, nChan,\
                        pressureLevels, pressureLayers, temperatureLayers, humidityLayers, ozoneConcLayers,\
                        surfaceType, surfaceTemperature, windSpeed10m, windDirection10m )
        crtmTb.append(Tb)

    tRttov = np.log(rttovObj.TauLevels[iprof, 999, 0::])
    dTauRttov =np.flipud(np.diff(np.flipud(tRttov)))
    ttRttov = np.exp(-1*np.cumsum(dTauRttov))
    num = rttovObj.TauLevels[iprof, 999, 1::] - rttovObj.TauLevels[iprof, 999, 0:nlevels-1] 
    den = np.log(myProfiles.P[iprof, 0:nlevels-1]) - np.log(myProfiles.P[iprof, 1:nlevels]) 
    tCrtm = np.exp(-1*np.cumsum(Transmission[999,0::]))
    numCrtm = np.diff(tCrtm[0::])
    print(ttRttov.shape)
    numRttov = ttRttov[1::]-ttRttov[0:ttRttov.shape[0]-1]
    plt.plot(numCrtm[1::]/den[1:len(numCrtm)],'rx')
    plt.plot(numRttov/den[0:len(numRttov)],'kx')
    plt.plot(num[1::]/den[1::],'bx')
    print('n rttov, ncrtm',numRttov.shape,numCrtm.shape)
    plt.savefig('whir.png')
    z = np.asarray(crtmTb)-rttovObj.Bt
    print(np.asarray(crtmTb).shape)
    for i,zz in enumerate(z[0]):
        print(i+1, zz,np.asarray(crtmTb)[0,i],rttovObj.Bt[0,i])

    for i,zz in enumerate(z[1]):
        print(i+1, zz,np.asarray(crtmTb)[1,i],rttovObj.Bt[1,i])
