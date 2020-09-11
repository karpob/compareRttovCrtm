#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import configparser, os, sys, h5py
import numpy as np
from lib.graphics.profile import plotContour, plotLines,plotContourLabelIdx
from lib.graphics.linePlots import basicLine
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
from lib.pycrtm.units import  waterPpmvDry2GmKgDry, gasPpmvMoistToDry
from lib.pycrtm.interpolation import profileInterpolate 
from matplotlib import pyplot as plt

def readProfileItemsH5( filename, additionalItems = []):
    """
    Read an RTTOV-style atmosphere profile.
    In: filename to hdf5
    Out: Pressure, Temperature, CO2, O3 [nprofiles,nlevels]
    Out: Gas_Units (mass, ppmv dry, ppmv moist)
    """
    items = ['T','Q','O3']
    if(len(additionalItems)>0):
        for i in additionalItems:
            items.append(i)
    h5 = h5py.File( filename )
    groups = list(h5['PROFILES'].keys())
    nprofiles = len(groups)
    nlevs, = np.asarray( h5['PROFILES'][groups[0]]['P'] ).shape 
    P = np.zeros([nprofiles,nlevs])
    itemsOut = {}
    for i in items: itemsOut[i] = np.zeros([nprofiles,nlevs])
    for i,g in enumerate(groups):
        P[i,:] = np.asarray(h5['PROFILES'][g]['P'])
        for ii in items:
            itemsOut[ii][i,:] = np.asarray(h5['PROFILES'][g][ii])
        GasUnits = int(np.asarray(h5['PROFILES'][g]['GAS_UNITS']))
    
    return P, itemsOut, GasUnits 

def setRttovProfiles( h5ProfileFileName, additionalItems=[] ):
    nlevels = 101  
    nprofiles = 6
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles.P, profileItems, myProfiles.GasUnits = readProfileItemsH5( h5ProfileFileName, additionalItems)
    for item in list(profileItems.keys()):
        exec("myProfiles.{} = profileItems['{}']".format(item,item))
       
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
        Ps = myProfiles.P[:,-1]
        Ts = myProfiles.T[:,-1]
        Qs = myProfiles.Q[:,-1]
        
        s2m.append([Ps[i], Ts[i] , Qs[i], 0, 0., 10000.])
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

def setProfilesCRTM(h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems = [], method='average'):
    nprofiles = 6
    profilesCRTM = profilesCreate( 6, 100, nAerosols=0, nClouds=0, additionalGases = additionalItems )
   
    Pi, profileItems, gas_units = readProfileItemsH5(h5_ppmv, additionalItems = additionalItems)
    interpOb = profileInterpolate(layerPressuresCrtm, Pi, profileItems)
    interpOb.interpProfiles(method=method) 
    profilesCRTM.P[:,:], profileItems = interpOb.get() 

    for i in list(profileItems.keys()):
        exec( "profilesCRTM.{}[:,:] = profileItems['{}']".format(i,i) )
    for i in list(profileItems.keys()):
        if ( i != 'T' ):
            exec('profilesCRTM.{}[:,:] = gasPpmvMoistToDry(profilesCRTM.{}[:,:], profilesCRTM.Q[:,:])'.format(i,i))
        
    profilesCRTM.Q[:,:] = waterPpmvDry2GmKgDry(profilesCRTM.Q[:,:]) 
    profilesCRTM.Pi[:,:] = Pi

    profilesCRTM.Angles[:,:] = 0.0
    profilesCRTM.Angles[:,2] = 100.0  # Solar Zenith Angle 100 degrees zenith below horizon.

    profilesCRTM.DateTimes[:,0] = 2015
    profilesCRTM.DateTimes[:,1] = 8
    profilesCRTM.DateTimes[:,2] = 1

    # Turn off Aerosols and Clouds
    #profilesCRTM.aerosolType[:] = -1
    #profilesCRTM.cloudType[:] = -1

    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 270.0 
    profilesCRTM.S2m[:,1] = 35.0 # just use salinity out of S2m for the moment.
    profilesCRTM.windSpeed10m[:] = 0.0
    profilesCRTM.windDirection10m[:] = 0.0 

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
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'cris-fsr_n20.TauCoeff.bin') )
    coefLevCrtm = np.asarray(crtmTauCoef['level_pressure'])
    layerPressuresCrtm = np.asarray(crtmTauCoef['layer_pressure'])

    ##########################
    # Set Profiles
    ##########################
    h5_mass =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas_kgkg.H5')
    h5_ppmv =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas.H5')

    profilesCRTM  = setProfilesCRTM( h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems=['CO2','CO'] )
    myProfiles = setRttovProfiles( h5_mass, additionalItems=['CO2','CO'])

    print("Now on to CRTM.")

    #########################
    # Run CRTM
    #########################
    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    crtmOb.sensor_id = 'cris-fsr_n20' 
    crtmOb.nThreads = 6

    crtmOb.loadInst()
    crtmOb.runK()

    #########################
    # End Run CRTM
    #########################

 

    # RT is done! Great! Let's make some plots!

    wv = np.asarray(crtmOb.Wavenumbers)
    profileNames = ['1 Tropical','2 Mid-Lat Summer', '3 Mid-Lat Winter', '4 Sub-Arctic Summer', '5 Sub-Arctic Winter', '6 US-Standard Atmosphere' ]
    sensitivities = ['O3','Q','T','CO2','CO']
    profileItems = {}
    for s in sensitivities:
        exec('profileItems[s] = myProfiles.{}'.format(s))

    plevs = np.zeros([6,coefLevCrtm.shape[0]])
    plevs[:,:] = coefLevCrtm
    interpOb = profileInterpolate(layerPressuresCrtm, plevs, profileItems) #coef levels are the same for rttov and crtm 
    interpOb.interpProfiles(method='crtm-wrap') 
    _, interpRttovProfileItems = interpOb.get() 

    idxLw = np.array([65,69,73,67,75,44,83,85,89,91,95,99,103,107,109,127,131,125,121,123,153])-1
    idxSw = np.array([1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959])-1
    idxSwLw = np.array([65,69,73,67,75,44,83,85,89,91,95,99,103,107,109,127,131,125,121,123,153,1939,1940,1941,1942,1943,1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959])-1
    for i,n in enumerate(profileNames): 
        key = n.replace(" ","_")+'_'
        s = 'Weighting Function'
        sValCrtm =-1.0*np.diff(crtmOb.TauLevels[i,:,:])/np.diff(np.log(profilesCRTM.P[i,:]))

        plotContour(wv, profilesCRTM.P[i,:], sValCrtm,\
                    'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',\
                    profileNames[i]+' CRTM {}'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                    key+'{}_crtm.png'.format( s.lower() ),\
                    zlim = [-1.0, 1.0], figureResolution=300 ) 
        wvTrunc = []
        for w in wv[idxSw]:
            wvTrunc.append('{0:.3f}'.format(w))
        plotContourLabelIdx(wvTrunc, profilesCRTM.P[i,:], sValCrtm[idxSw],\
                    'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',\
                    profileNames[i]+' CRTM {}'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                    key+'{}_crtm_sw.png'.format( s.lower() ),\
                    zlim = [-1.0,1.0],\
                    figureResolution=300 ) 
        wvTrunc = []
        for w in wv[idxLw]:
            wvTrunc.append('{0:.3f}'.format(w))
        plotContourLabelIdx(wvTrunc, profilesCRTM.P[i,:], sValCrtm[idxLw],\
                    'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',\
                    profileNames[i]+' CRTM {}'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                    key+'{}_crtm_lw.png'.format( s.lower() ),\
                    zlim = [-1.0,1.0],\
                    figureResolution=300 ) 
        for ii in range(idxLw.shape[0]):
            idxPair = np.array([idxLw[ii],idxSw[ii]]) 
            nP = profilesCRTM.P[i,:].shape[0]
            plotLines ( sValCrtm[idxPair], profilesCRTM.P[i,0:nP-1], "Weighting Function", \
                   "Pressure [hPa]",(idxPair+1).tolist(),\
                   'Weighting Functions LW/SW pair', key+'{}_crtm_sub_{}_{}_lines.png'.format( s.lower(),idxLw[ii]+1,idxSw[ii]+1 ),\
                    ylim =[100,1000],figureResolution = 300 )


        for s in sensitivities:
            #exec('sValCrtm = crtmOb.{}K[i,:,:]*profilesCRTM.{}[i,:]'.format(s,s)) 
            exec('sValCrtm = crtmOb.{}K[i,:,:]'.format(s,s)) 
            maxS = sValCrtm.max().max()
            minS = sValCrtm.min().min()
            symMaxS = max(abs(minS),abs(maxS))
            symMinS = -1.0*symMaxS


            maxS2 = sValCrtm[idxSwLw].max().max()
            minS2 = sValCrtm[idxSwLw].min().min()
            symMaxS2 = max(abs(minS2),abs(maxS2))
            symMinS2 = -1.0*symMaxS2
           
            plotContour(wv, profilesCRTM.P[i,:], sValCrtm,\
                        'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [dTb/dx]',\
                        profileNames[i]+' CRTM {} Jacobian'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                        key+'{}k_crtm.png'.format( s.lower() ),\
                        zlim = [symMinS, symMaxS], figureResolution=300 ) 
            wvTrunc = []
            for w in wv[idxSw]:
                wvTrunc.append('{0:.3f}'.format(w))
            plotContourLabelIdx(wvTrunc, profilesCRTM.P[i,:], sValCrtm[idxSw],\
                        'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [dTb/dx]',\
                        profileNames[i]+' CRTM {} Jacobian'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                        key+'{}k_crtm_sw.png'.format( s.lower() ),\
                        zlim = [symMinS2,symMaxS2],\
                        figureResolution=300 ) 
            wvTrunc = []
            for w in wv[idxLw]:
                wvTrunc.append('{0:.3f}'.format(w))
            plotContourLabelIdx(wvTrunc, profilesCRTM.P[i,:], sValCrtm[idxLw],\
                        'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [dTb/dx]',\
                        profileNames[i]+' CRTM {} Jacobian'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                        key+'{}k_crtm_lw.png'.format( s.lower() ),\
                        zlim = [symMinS2,symMaxS2],\
                        figureResolution=300 ) 
            for ii in range(idxLw.shape[0]):
                idxPair = np.array([idxLw[ii],idxSw[ii]]) 
                plotLines ( sValCrtm[idxPair], profilesCRTM.P[i,:], "Jacobian [dTb/dx]", \
                       "Pressure [hPa]",(idxPair+1).tolist(),\
                       'Jacobians', key+'{}k_crtm_sub_{}_{}_lines.png'.format( s.lower(),idxLw[ii]+1,idxSw[ii]+1 ),\
                        ylim =[100,1000],figureResolution = 300 )
 
