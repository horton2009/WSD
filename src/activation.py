#!/usr/bin/python

"""Activation Function Classes.

Provides some of the most common activation functions to be used by
hidden or output neurons.
"""

__author__       = 'Sam Gibson'
__version__      = '0.1'
__lastmodified__ = 'Tue Feb  8 23:27:57 PST 2005'

import math

class Activation :
   """Base activation class. All activations should be derived from
   this. To implement your own activation, simply create a derivative
   class and implement class.activate( self, neti ) where `neti' is
   the net input to the neuron."""
   
   def activate ( self, neti ) :			# neti = Ei (Wi*Xi)
     pass
     
"""TODO : Document each of these."""

class BinaryThreshold (Activation) :		 # return 0 / 1			
   theta = 0.0

   def activate ( self, neti ) :
      if ( neti < self.theta ) : return 0.0
      else : return 1.0

class Linear (Activation) :			 # return self
   def activate ( self, neti ) :
      return neti

class Sigmoid (Activation) :
   theta = 0.0
   tau = 1.0

   def activate ( self, neti ) :
      return (1 / (1 + math.exp((-1 * neti - self.theta) / self.tau)))		# return  1/(1+e-neti)

class TanH (Activation) :
   lamda = 1.0

   def activate ( self, neti ) :
      return (0.5 * (1 + math.tanh(self.lamda * neti)))			# math.tanh(): return -1~1 ==> (1+math.tanh())/2 : return 0~1

class Gaussian (Activation) :
   sigma = 0.159155

   def activate ( self, neti ) :
      return math.exp((neti - 1) / (self.sigma * self.sigma))			# wrong: f=exp(-((x-u)**2/sigma**2)) : return 0~1 







