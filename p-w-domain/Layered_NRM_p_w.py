#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:56:47 2018

@author: christianreini
"""

from Wavefield_NRM_p_w import Wavefield_NRM_p_w
import numpy as np
import sys

class Layered_NRM_p_w(Wavefield_NRM_p_w):
    
    def __init__(self,nt,dt,nr=1,dx=1,verbose=0,avec=np.zeros(1),bvec=np.zeros(1),g1vec=np.zeros(1),g3vec=np.zeros(1),p1=None,ReciprocalMedium=False):
        """
        defines a 1D (non-)reciprocal medium, and initialises a wavefield associated with a single ray-parameter in the ray-parameter frequency domain.
        
        The class Layered_NRM_p_w defines a 1.5D (non-)reciprocal medium and a scalar wavefield. We consider a single horizontal ray-parameter p1 and all frequencies that are sampled by the given number of time samples nt and the time sample interval dt.
        
        **Data format**:
         - Wavefields are saved in an array of dimensions (nt,nr).
         - Wavefields are in the p-`\\omega` domain.
         - The zero frequency component is placed at the first index position.
         - If the wavefield is transformed to the time domain, the zero time component is placed at the center of the time dimension.
        
        :param nt: Number of time samples
        :param dt: Time sample interval in seconds
        :param nr: Number of space samples
        :param dx: Space sample interval
        :param verbose: Set verbose=1 to receive feedback in the command line.
        :param avec: Medium parameter`\\alpha` (real-valued), must have the shape (n,) for n layers
        :param bvec: Medium parameter`\\beta` (real-valued), must have the shape (n,) for n layers
        :param g1vec: Medium parameter`\\gamma_1`  (real-valued?), must have the shape (n,) for n layers
        :param g3vec: Medium parameter`\\gamma_3`  (real-valued?), must have the shape (n,) for n layers
        :param p1: Horizontal ray-parameter in seconds per metre
        :param ReciprocalMedium: For non-reciprocal media set ReciprocalMedium=False, for reciprocal media set ReciprocalMedium=True 
        :type nt: int
        :type dt: int, float
        :type nr: int
        :type dx: int, float
        :type verbose: int
        :type avec: numpy.ndarray (real-valued)
        :type bvec: numpy.ndarray (real-valued)
        :type g1vec: numpy.ndarray (real-valued?)
        :type g3vec: numpy.ndarray (real-valued?)
        :type p1: int, float
        :type ReciprocalMedium: bool
        
        :Example:

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        >>> F=LM(nt=1024,dt=0.005,avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),p1=2e-4,ReciprocalMedium=False)
        
        .. csv-table:: Unified scalar wavefields: Quantities
        :header: "Wavefield", "P", "`Q_1`", "`Q_3`", "`\\alpha`", "`\\beta`", "`\\gamma_1`", "`\\gamma_3`", "`\\delta_1`", "`\\delta_3`", "B", "`C_1`", "`C_3`" 
        :widths: 15, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5

       "TE", ,,,,,,,,,,,
       "TM", ,,,,,,,,,,,
       "Ac. (fluid)", "`p`", "`v_1`", "`v_3`", "`\\kappa`", "`\\rho`", "`d_1`", "`d_3`", "`e_1`", "`e_3`", "`q`", "`f_1`", "`f_3`"
       "SH (solid)", ,,,,,,,,,,,
        """
        
        # Inherit __init__ from Wavefield_NRM_p_w
        Wavefield_NRM_p_w.__init__(self,nt,dt,nr,dx)
        
        # Check if medium parameters are passed as arrays
        if not ( isinstance(avec,np.ndarray) and isinstance(bvec,np.ndarray) and isinstance(g1vec,np.ndarray) and isinstance(g3vec,np.ndarray) ):
            sys.exit('avec, bvec, g1vec and g3vec have to be of the type numpy.ndarray.')
            
        # Set gamma_1 and gamma_3 by default equal to zero
        if g1vec.all() == np.zeros(1):
            g1vec = np.zeros_like(avec)
        if g3vec.all() == np.zeros(1):
            g3vec = np.zeros_like(avec)
            
        # Force the medium parameters to have identical shape
        if avec.shape!=bvec.shape or avec.shape!=g1vec.shape or avec.shape!=g3vec.shape:
            sys.exit('avec, bvec, g1vec and g3vec have to be of identical shape.')
        
        # Force the medium parameters to be 1-dimensional, i.e. e.g. avec.shape=(n,)
        if avec.ndim!=1:
            sys.exit('avec.ndim, bvec.ndim, g1vec.ndim and g3vec.ndim must be one.')
            
        # Check if medium parameters correspond to a lossless (non-)reciprocal medium
        if ReciprocalMedium == False:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.imag.any()!=0 or g3vec.imag.any()!=0:
                sys.exit('In lossless non-reciprocal media the imaginary value of avec, bvec, g1vec and g3vec has to be zero.')
        elif ReciprocalMedium == True:
            if avec.imag.any()!=0 or bvec.imag.any()!=0 or g1vec.real.any()!=0 or g3vec.real.any()!=0:
                sys.exit('In lossless reciprocal media the imaginary value of avec and bvec has to be zero, the real value of g1vec and g3vec has to be zero.')
            
        # Set medium parameters    
        self.avec = avec
        self.bvec = bvec
        self.g1vec = g1vec
        self.g3vec = g3vec
        self.p1 = p1
        self.ReciprocalMedium = ReciprocalMedium
        
        # Calculate vertical ray-parameter p3 
        # Note: By default python uses opposite sign convention for evanescent waves as Kees': (-1)**0.5=1j
        p3 = np.zeros(np.shape(self.avec),dtype=complex)
        if self.ReciprocalMedium is True:
            p3[:] = self.avec*self.bvec + self.g1vec**2 + self.g3vec**2 - self.p1**2
        elif self.ReciprocalMedium is False:
            p3[:] = self.avec*self.bvec - self.g1vec**2 + 2*self.g1vec*self.p1 - self.p1**2
        self.p3 = p3**0.5
        
    def L_eigenvectors_p_w(self,beta=None,g3=None,p3=None,normalisation='flux'):
        """
        computes the eigenvector matrix L and its inverse Linv, either in flux- or in pressure-normalisation.
        
        :param beta: Medium parameter`\\beta`  (real-valued), must be a scalar
        :param g3: Medium parameter`\\gamma_3`  (real-valued?), must be a scalar
        :param p3: Vertical ray-parameter `p_3`, must be a scalar
        :type beta: int, float
        :type g3: int, float
        :type p3: int, float
        :type normalisation: str
        :return: L, numpy.ndarray (2,2)
        :return: Linv, numpy.ndarray (2,2)
        :rtype: dict
        
        :Example:

        >>> from Layered_NRM_p_w import Layered_NRM_p_w as LM
        >>> import numpy as np
        >>> F=LM(nt=1024,dt=0.005,avec=np.array([1,2,3]),bvec=np.array([0.4,3.14,2]),g1vec=np.array([0.9,2.1,0.3]),g3vec=np.array([0.7,1.14,0.2]),p1=2e-4,ReciprocalMedium=False)
        >>> Lvecs=F.L_eigenvectors_p_w(beta=0.1,g3=0.4,p3=2e-4,normalisation='flux')
        >>> L = Lvecs['L']
        >>> Linv = Lvecs['Linv']
        
        .. note:: The eigenvector matrix L and its inverse Linv are different for reciprocal and non-reciprocal media.
        """
        # Check if required input variables are given
        if (beta is None) or (g3 is None) or (p3 is None):
            sys.exit('The input variables \'beta\', \'g3\' and  \'p3\' of the function L_eigenvectors_p_w must be set.')
         
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('The input variable \'normalisation\' of the function L_eigenvectors_p_W must be set, either to \'flux\', or to \'pressure\'.')
            
        # Initialise L and Linv
        L = np.zeros((2,2),dtype=complex)
        Linv = np.zeros((2,2),dtype=complex)
            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            # L matrix
            L[0,0] = beta/p3
            L[0,1] = beta/p3
            L[1,0] = p3/beta
            L[1,1] = -p3/beta
            L = (L/2)**0.5
            
            # Inverse L matrix
            Linv[0,0] = p3/beta
            Linv[0,1] = beta/p3
            Linv[1,0] = p3/beta
            Linv[1,1] = -beta/p3
            Linv = (Linv/2)**0.5
            
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = p3/beta
            L[1,1] = -p3/beta
            
            # Inverse L matrix
            Linv[0,0] = 1
            Linv[0,1] = beta/p3
            Linv[1,0] = 1
            Linv[1,1] = -beta/p3
            Linv = 0.5*Linv
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = (p3+g3)/beta
            L[1,1] = -(p3-g3)/beta
            
            # Inverse L matrix
            Linv[0,0] = (p3-g3)/beta
            Linv[0,1] = 1
            Linv[1,0] = (p3+g3)/beta
            Linv[1,1] = -1
            
            fac = (beta/(2*p3))**0.5
            L = fac*L
            Linv = fac*Linv
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            # L matrix
            L[0,0] = 1
            L[0,1] = 1
            L[1,0] = (p3+g3)/beta
            L[1,1] = -(p3-g3)/beta
            
            # Inverse L matrix
            Linv[0,0] = (p3-g3)/beta
            Linv[0,1] = 1
            Linv[1,0] = (p3+g3)/beta
            Linv[1,1] = -1
            Linv = beta/(2*p3)*Linv
         
        out = {'L':L,'Linv':Linv}
        return out
          
    def RT_p_w(self,beta_u=None,g3_u=None,p3_u=None,beta_l=None,g3_l=None,p3_l=None,normalisation='flux'):
        
        # Check if required input variables are given
        if (beta_u is None) or (g3_u is None) or (p3_u is None) or (beta_l is None) or (g3_l is None) or (p3_l is None):
            sys.exit('The input variables \'beta_u\', \'g3_u\',  \'p3_u\', \'beta_l\', \'g3_l\',  \'p3_l\' of the function RT_p_w must be set.')
         
        # Check if normalisation is set correctly
        if (normalisation is not 'flux') and (normalisation is not 'pressure'):
            sys.exit('The input variable \'normalisation\' of the function RT_p_w must be set, either to \'flux\', or to \'pressure\'.')
            
        if (self.ReciprocalMedium is True) and (normalisation is 'flux'):
            rP = (p3_u*beta_l - p3_l*beta_u) / (p3_u*beta_l + p3_l*beta_u)
            rM = -rP
            tP = 2*(p3_u*beta_l*p3_l*beta_u)**0.5
            tM = tP
        
        elif (self.ReciprocalMedium is True) and (normalisation is 'pressure'):
            rP = (p3_u*beta_l - p3_l*beta_u) / (p3_u*beta_l + p3_l*beta_u)
            rM = -rP
            tP = 2*p3_u*beta_l/(p3_u*beta_l+p3_l*beta_u)
            tM = 1 - rP
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'flux'):
            rP = ((p3_u+g3_u)*beta_l-(p3_l+g3_l)*beta_u) / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            rM = -((p3_u-g3_u)*beta_l-(p3_l-g3_l)*beta_u) / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            tP = 2*(p3_u*beta_l*p3_l*beta_u)**0.5
            tM = tP
            
        elif (self.ReciprocalMedium is False) and (normalisation is 'pressure'):
            rP = ((p3_u+g3_u)*beta_l-(p3_l+g3_l)*beta_u) / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            rM = -((p3_u-g3_u)*beta_l-(p3_l-g3_l)*beta_u) / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            tP = 2*p3_u*beta_l / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            tM = 2*p3_l*beta_u / ((p3_u-g3_u)*beta_l+(p3_l+g3_l)*beta_u)
            
        out = {'rP':rP,'tP':tP,'rM':rM,'tM':tM}
        return out