#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import configparser, os, sys, h5py
import numpy as np
from lib.graphics.profile import plotContour, plotLines
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
from matplotlib import pyplot as plt

def interpolateProfile(x, xo, yo):
    """
    Do a log-linear interpolation.
    """
    logX =  np.log(x)
    logXo = np.log(xo)
    logYo = np.log(yo)
    return np.exp(np.interp(logX, logXo, logYo))
 
def interpProfiles( Po, Pi, Ti, Qi, CO2i, O3i):
    """
    Interpolate log-linear variables(Pi,Ti,Qi,CO2i,O3i) onto new pressure grid (Po)
    """
    nlevs = Po.shape[0]
    nprofiles = Pi.shape[0]
    To = np.zeros([nprofiles,nlevs])
    Qo = np.zeros([nprofiles,nlevs])
    CO2o = np.zeros([nprofiles,nlevs])
    O3o = np.zeros([nprofiles,nlevs])
    Poo = np.zeros([nprofiles,nlevs])
    for i in list(range(nprofiles)):
        Poo[i,:]  = np.asarray(Po)[:]
        O3o[i,:]  = interpolateProfile( Po, Pi[i,:], O3i[i,:]  ) 
        CO2o[i,:] = interpolateProfile( Po, Pi[i,:], CO2i[i,:] ) 
        Qo[i,:]   = interpolateProfile( Po, Pi[i,:], Qi[i,:]   ) 
        To[i,:]   = interpolateProfile( Po, Pi[i,:], Ti[i,:]   ) 
    return Poo, To, Qo, CO2o, O3o

def readProfileH5( filename ):
    """
    Read an RTTOV-style atmosphere profile.
    In: filename to hdf5
    Out: Pressure, Temperature, CO2, O3 [nprofiles,nlevels]
    Out: Gas_Units (mass, ppmv dry, ppmv moist)
    """
    h5 = h5py.File( filename )
    groups = list(h5['PROFILES'].keys())
    nprofiles = len(groups)
    nlevs, = np.asarray( h5['PROFILES'][groups[0]]['P'] ).shape 
    P = np.zeros([nprofiles,nlevs])
    T = np.zeros([nprofiles,nlevs])
    Q = np.zeros([nprofiles,nlevs])
    CO2 = np.zeros([nprofiles,nlevs])
    O3 = np.zeros([nprofiles,nlevs])

    for i,g in enumerate(groups):
        P[i,:] = np.asarray(h5['PROFILES'][g]['P'])
        Q[i,:] = np.asarray(h5['PROFILES'][g]['Q'])
        T[i,:] = np.asarray(h5['PROFILES'][g]['T'])
        CO2[i,:] = np.asarray(h5['PROFILES'][g]['CO2'])
        O3[i,:] = np.asarray(h5['PROFILES'][g]['O3'])
        GasUnits = int(np.asarray(h5['PROFILES'][g]['GAS_UNITS']))
    
    return P, T, Q, CO2, O3, GasUnits 

def setRttovProfiles( h5ProfileFileName):
    nlevels = 101  
    nprofiles = 6
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5(h5ProfileFileName)
    myProfiles2 = pyrttov.Profiles(nprofiles,nlevels-1)
    myProfiles2.P, myProfiles2.T, myProfiles2.Q, myProfiles2.CO2, myProfiles2.O3 =  interpProfiles( layerPressuresCrtm, myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3)
    myProfiles2.GasUnits = myProfiles.GasUnits
    myProfiles = pyrttov.Profiles(nprofiles,nlevels-1)
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
        s2m.append([myProfiles.P[i,-1], myProfiles.T[i,-1] ,myProfiles.Q[i,-1] , 0, 0., 10000.])
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
    return myProfiles

def setProfilesCRTM(h5_mass, h5_ppmv):

    nprofiles = 6
    myProfiles = pyrttov.Profiles(nprofiles, 101)
    profilesCRTM = profilesCreate( 6, 100 )
    myProfiles.P, myProfiles.T, myProfiles.Q, myProfiles.CO2, myProfiles.O3, myProfiles.GasUnits = readProfileH5( h5_mass )
    _, _, _, CO2_1, O3_1, units_1 = readProfileH5( h5_ppmv )
    crtmTauCoef = [] # clear out some ram. by getting rid of the dictonary and set it to empty list
    profilesCRTM.P[:,:], profilesCRTM.T[:,:], profilesCRTM.Q[:,:], profilesCRTM.CO2[:,:], profilesCRTM.O3[:,:] =  interpProfiles( layerPressuresCrtm, myProfiles.P, myProfiles.T, 1000.0*myProfiles.Q, CO2_1, O3_1)
    profilesCRTM.Angles[:,:] = 0.0
    profilesCRTM.Angles[:,2] = 100.0  # Solar Zenith Angle 100 degrees zenith below horizon.

    profilesCRTM.DateTimes[:,0] = 2015
    profilesCRTM.DateTimes[:,1] = 8
    profilesCRTM.DateTimes[:,2] = 1

    profilesCRTM.Pi[:,:] = coefLevCrtm

    # Turn off Aerosols and Clouds
    profilesCRTM.aerosolType[:] = -1
    profilesCRTM.cloudType[:] = -1

    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 270 
    profilesCRTM.S2m[:,1] = 35.0 # just use salinity out of S2m for the moment.
    profilesCRTM.windSpeed10m[:] = 0.0
    profilesCRTM.windDirection10m[:] = 0.0 
    profilesCRTM.n_absorbers[:] = 3 

    # land, soil, veg, water, snow, ice
    profilesCRTM.surfaceTypes[:,3] = 1 

    return profilesCRTM
if __name__ == "__main__":
    #################################################################################################
    # Get installed path to coefficients from pycrtm submodule install (crtm.cfg in pycrtm directory)
    # load stuff we need for CRTM coefficients
    #################################################################################################
    pathToThisScript = os.path.dirname(os.path.abspath(__file__))
    pathInfo = configparser.ConfigParser()
    pathInfo.read( os.path.join(pathToThisScript,'lib','pycrtm','crtm.cfg') )
    coefficientPathCrtm = pathInfo['CRTM']['coeffs_dir']
    #################################################################################################
    # Get CRTM coefficient interface levels, and pressure layers 
    # Pull pressure levels out of coefficient
    # get pressures used for profile training in CRTM.
    #################################################################################################
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'iasi_metop-b.TauCoeff.bin') )
    coefLevCrtm = np.asarray(crtmTauCoef['level_pressure'])
    layerPressuresCrtm = np.asarray(crtmTauCoef['layer_pressure'])
   
    ##########################
    # Set Profiles
    ##########################
    h5_mass =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas_kgkg.H5')
    h5_ppmv =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas.H5')

    myProfiles = setRttovProfiles( h5_ppmv )
    profilesCRTM  = setProfilesCRTM( h5_mass, h5_ppmv )

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
    #rttovObj.Options.UseQ2m = True
    rttovObj.Options.DoNlteCorrection = True
    rttovObj.Options.AddSolar = True
    rttovObj.Options.FastemVersion = int(6)
    rttovObj.Options.Nthreads = 6
    rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov9pred101L','rtcoef_metop_2_iasi.H5')

    #load the coefficient
    rttovObj.loadInst()

    #load the profiles
    rttovObj.Profiles = myProfiles

    # have rttov calculate surface emission.
    rttovObj.SurfEmisRefl = -1.0*np.ones((2,myProfiles.P.shape[0],rttovObj.Nchannels), dtype=np.float64)
    
    # run it 
    rttovObj.runK()
    
    ########################
    # End Run RTTOV
    ########################

    print("Now on to CRTM.")
    

    #########################
    # Run CRTM
    #########################
    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    crtmOb.sensor_id = 'iasi_metop-b' 
    crtmOb.nThreads = 6

    crtmOb.loadInst()
    crtmOb.runK()

    #########################
    # End Run CRTM
    #########################


    # RT is done! Great! Let's make some plots!

    # get the 616 channel subset for IASI
    h5 = h5py.File('iasi_wavenumbers.h5')
    idx =np.asarray( h5['idxBufrSubset'])-1 
    wv = np.asarray(crtmOb.Wavenumbers)[idx]

    profileNames = ['1 Tropical','2 Mid-Lat Summer', '3 Mid-Lat Winter', '4 Sub-Arctic Summer', '5 Sub-Arctic Winter', '6 US-Standard Atmosphere' ]
    for i,n in enumerate(profileNames): 
        key = n.replace(" ","_")+'_'
        o3SenCrtm = crtmOb.O3K[i,idx,:]*profilesCRTM.O3[i,:]
        qSenCrtm = crtmOb.QK[i,idx,:]*profilesCRTM.Q[i,:] 
        tSenCrtm = crtmOb.TK[i,idx,:]*profilesCRTM.T[i,:]   
        co2SenCrtm =  crtmOb.CO2K[i,idx,:]*profilesCRTM.CO2[i,:]

        o3SenRttov = rttovObj.O3K[i,idx,:]*myProfiles.O3[i,:]
        qSenRttov = rttovObj.QK[i,idx,:]*myProfiles.Q[i,:]
        tSenRttov =  rttovObj.TK[i,idx,:]*myProfiles.T[i,:]
        co2SenRttov =  rttovObj.CO2K[i,idx,:]*myProfiles.CO2[i,:]

        maxO3 = max(o3SenCrtm.max().max(),o3SenRttov.max().max())
        minO3 = min(o3SenCrtm.min().min(),o3SenRttov.min().min())

        maxQ = max(qSenCrtm.max().max(),qSenRttov.max().max())
        minQ = min(qSenCrtm.min().min(),qSenRttov.min().min())

        maxT = max(tSenCrtm.max().max(),tSenRttov.max().max())
        minT = min(tSenCrtm.min().min(),tSenRttov.min().min())

        maxCo2 = max(co2SenCrtm.max().max(),co2SenRttov.max().max())
        minCo2 = min(co2SenCrtm.min().min(),co2SenRttov.min().min())
   
        wfCRTM =-1.0*np.diff(crtmOb.TauLevels[i,idx,:])/np.diff(np.log(profilesCRTM.P[i,:]))
        wfRTTOV =-1.0* np.diff(rttovObj.TauLevels[i,idx,:])/np.diff(np.log(myProfiles.P[i,:]))

        maxWf = max(wfCRTM.max().max(),wfRTTOV.max().max())
        minWf = min(wfCRTM.min().min(),wfRTTOV.min().min())

        plotContour(wv,profilesCRTM.P[i,:],o3SenCrtm,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]', profileNames[i]+' CRTM O$_3$ Jacobian',key+'o3k_crtm.png', zlim = [minO3, maxO3])    
        plotContour(wv,profilesCRTM.P[i,:],qSenCrtm,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' CRTM H$_2$O Jacobian',key+'h2ok_crtm.png', zlim = [minQ, maxQ])    
        plotContour(wv,profilesCRTM.P[i,:],tSenCrtm,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' CRTM T Jacobian',key+'Tk_crtm.png', zlim = [minT, maxT])    
        plotContour(wv,profilesCRTM.P[i,:],co2SenCrtm,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' CRTM CO$_2$ Jacobian',key+'co2_crtm.png', zlim = [minCo2, maxCo2])    
        plotContour(wv,profilesCRTM.P[i,:],wfCRTM[:,:],'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',profileNames[i]+' CRTM Weighting Function',key+'WF_crtm.png', zlim = [minWf, maxWf])    
        
        plotContour(wv, myProfiles.P[i,:],o3SenRttov,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' RTTOV O$_3$ Jacobian',key+'o3k_rttov.png', zlim = [minO3, maxO3])    
        plotContour(wv, myProfiles.P[i,:],qSenRttov,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+'  RTTOV H$_2$O Jacobian',key+'h2ok_rttov.png', zlim = [minQ, maxQ])    
        plotContour(wv, myProfiles.P[i,:], tSenRttov,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' RTTOV T Jacobian',key+'Tk_rttov.png', zlim = [minT, maxT])    
        plotContour(wv, myProfiles.P[i,:], co2SenRttov,'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',profileNames[i]+' RTTOV CO$_2$ Jacobian',key+'co2_rttov.png', zlim = [minCo2, maxCo2])    
        plotContour(wv, myProfiles.P[i,:], wfRTTOV[:,:],'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',profileNames[i]+' RTTOV Weighting Function',key+'WF_rttov.png', zlim = [minWf, maxWf])    

        plt.figure()
        plt.plot(wv, crtmOb.Bt[i,idx],'b',label='CRTM')
        plt.plot(wv, rttovObj.Bt[i,idx],'r',label='RTTOV')
        plt.xlabel('Wavenumber [cm$^{-1}$]')
        plt.ylabel('Brightness Temperature [K]')
        plt.title('IASI Brightness Temperature Profile '+n)
        plt.legend()
        plt.savefig(key+'iasi_crtm_rttov.png')
        plt.close()

        plt.figure()
        plt.plot(wv, crtmOb.Bt[i,idx]-rttovObj.Bt[i,idx],'k')
        plt.title('IASI Brightness Temperature Difference Profile '+n)
        plt.xlabel('Wavenumber [cm$^{-1}$]')
        plt.ylabel('CRTM - RTTOV Brightness Temperature [K]')
        plt.savefig(key+'iasi_crtm_rttov_diff.png')
        plt.close()

        plt.figure()
        plt.plot(wv, crtmOb.surfEmisRefl[i,idx],'b', label='CRTM')
        plt.plot(wv, rttovObj.SurfEmisRefl[0,i,idx],'r', label='RTTOV')
        plt.legend()
        plt.title('IASI Emissivity Profile '+n)
        plt.xlabel('Wavenumber [cm$^{-1}$]')
        plt.ylabel('Emissivity')
        plt.savefig(key+'iasi_emissivity_crtm_rttov.png')
        plt.close()

        plt.figure()
        plt.plot(wv, crtmOb.surfEmisRefl[i,idx]- rttovObj.SurfEmisRefl[0,i,idx],'k')
        plt.title('IASI Emissivity Difference Profile '+n)
        plt.xlabel('Wavenumber [cm$^{-1}$]')
        plt.ylabel('CRTM - RTTOV Emissivity')
        plt.savefig(key+'iasi_emissivity_crtm_rttov_diff.png')
        plt.close()
