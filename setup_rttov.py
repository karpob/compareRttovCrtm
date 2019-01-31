#!/usr/bin/env python3
import os, shutil, glob, sys, argparse, tarfile, stat
from subprocess import Popen, PIPE

def main( a ):
    arch = a.arch
    installPath = a.install
    tarballPath = a.rtpath

    fo = open('rttov_install.stdo','wb')
    fe = open('rttov_install.stde','wb')

    scriptDir = os.path.split(os.path.abspath(__file__))[0]
    if (arch == 'ifort-openmp'):
        os.environ['FC']='ifort'
        os.environ['CC']='icc'
        os.environ['CXX']='icpc'
    
    # check if rttov directory in install Path exists. If it exists delete and recreate, otherwise make it.
    if( os.path.isdir( os.path.join(a.install,'rttov') ) ):
        shutil.rmtree( os.path.join(a.install,'rttov') )
        os.mkdir( os.path.join(a.install,'rttov') )
    else:
        os.makedirs(os.path.join(a.install,'rttov') )
    
    print("Untarring rttov")
    # untar rttov
    if(a.version =='latest'):
        print(tarballPath)
        rttovVersions = glob.glob( os.path.join(tarballPath,'rttov*.tar.gz'))
        rttovVersions.sort()
        print(rttovVersions)
        rttovSelected = rttovVersions[-1]
    else:  rttovSelected = glob.glob( os.path.join(tarballPath,'rttov'+a.version+'*.tar.gz'))[-1]

    print(rttovSelected)
    t=tarfile.open(rttovSelected) 
    t.extractall(path=os.path.join(installPath,'rttov'))
    t.close()

    print("Running 2to3 so RTTOV interface is python3 compliant.")
    p=Popen(['2to3','-w','-n',os.path.join(installPath,'rttov','wrapper','pyrttov')],stdout=fo,stderr=fe)
    p.wait()
    checkProcess(p, 'RTTOV python3 patch.', fo, fe, scriptDir)     
    
    os.chdir(scriptDir)
    print("moving Makefile.local to rttov build")
    shutil.copy(os.path.join(scriptDir,'etc','Makefile.local'), os.path.join(installPath,'rttov','build'))
    os.chdir(os.path.join(installPath,'rttov','src'))
    os.environ['HDF5_PREFIX'] = a.hdf5path 

    print("compiling RTTOV")
    # run compile script selecting all desired options 
    #( echo gfortran-openmp; echo ./; echo y; echo y; echo -j5; echo y ) | ../build/rttov_compile.sh
    # get around the stupid script, and just go for it with the two commands you need to generate makefiles and compile.
    p= Popen(['../build/Makefile.PL','RTTOV_HDF=1','RTTOV_F2PY=1'],stdout=fo,stderr=fe)
    p.wait()
    checkProcess(p, 'Generate RTTOV makefiles', fo, fe, scriptDir)     

    p = Popen(['make','ARCH='+arch,'INSTALL_DIR=./','-j{}'.format(a.jproc)], stdout=fo, stderr=fe)
    p.wait()
    checkProcess(p, 'Compiling RTTOV', fo, fe, scriptDir)
    print("Done RTTOV compile.")     
    # move shared object to where rttov library expects it to be with the name it wants.
    os.chdir(os.path.join(installPath,'rttov'))

    # for rttov < 12.2
    if ( not  os.path.exists( os.path.join('lib','rttov_wrapper_f2py.so') ) ):
        shutil.copy( glob.glob( os.path.join( 'tmp-'+arch,'wrapper','rttov_wrapper_f2py*.so'))[0], os.path.join('lib','rttov_wrapper_f2py.so') ) 

     
    # Add line so simulators know where to look for RTTOV
    print("Modifying rtOptions.cfg")
    modifyOptionsCfg( 'rttov.cfg', scriptDir, installPath )
    print("For more information about the install look in {}, and {}".format(fo.name,fe.name) ) 
    print("Done!")

def checkProcess(p, name, fo, fe, scriptDir):
    if(p.returncode>0):
        foname = fo.name
        fename  = fe.name

        with open(os.path.join(scriptDir,foname),'r') as foOb:
            for l in foOb.readlines():
                print( l.strip() )
      
        with open(os.path.join(scriptDir,fename),'r') as feOb:
            for l in feOb.readlines():
                print( l.strip() )

        print("For more information about the install look in {}, and {}".format(fo.name,fe.name) ) 
        fo.close()
        fe.close()
        sys.exit(name+" failed.")

def modifyOptionsCfg( filename, scriptDir, installPath ):
    with open( filename ,'w') as newFile:
        with open(os.path.join(scriptDir, filename), 'r') as oldFile:
            for l in oldFile:
                if('rttov_install_dir' in l):
                    newFile.write(l.replace(l,'rttov_install_dir = '+os.path.join(installPath,'rttov')+os.linesep))
                else:
                    newFile.write(l)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = 'install crtm and pycrtm')
    parser.add_argument('--install',help = 'install path.', required = True, dest='install')
    parser.add_argument('--rtpath',help = 'path to RT tarballs.', required = True, dest='rtpath')
    parser.add_argument('--jproc',help = 'Number of threads to pass to make.', required = True, dest='jproc')
    parser.add_argument('--arch',help = 'compiler/architecture.', required = False, dest='arch', default='gfortran-openmp')
    parser.add_argument('--version',help = 'select which version, if there are multiple choices.', required = False, dest='version', default='latest')
    parser.add_argument('--hdf5-path',help = 'path to hdf5 library.', required = False, dest='hdf5path', default='latest')
    a = parser.parse_args()

    main( a )

