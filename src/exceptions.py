#!/usr/bin/python

"""Neural Exceptions

Defines a set of exceptions to be used by PyLibNeural."""

__author__       = 'Sam Gibson'
__version__      = '0.1'
__lastmodified__ = 'Thu Mar  3 12:11:26 PST 2005'

ERR_INPUTNONE    = 'Input is required, but is None.'
ERR_INPUTTYPE    = 'Invalid input type.'
ERR_VECTOR       = 'Vector is invalid or incorrect size.'
ERR_PARAMTYPE    = 'Parameter type is invalid.'
ERR_PARAMVAL     = 'Parameter is invalid.'
ERR_LAYERNONE    = 'Layer is invalid or None.'
ERR_LAYERNOHID   = 'Network has no hidden layers.'
ERR_INDEX        = 'Index is out of range or invalid'
ERR_DELTANONE    = 'Error has not been computed yet, or is invalid.'
ERR_NETWORKTYPE  = 'Invalid network type or configuration.'

class NeuralException (Exception) :
    """Encapsulate an ANN error or warning. May contain
    basic error or warning information. You may subclass it to 
    provide additional functionality, or to add localization."""

    def __init__ ( self, msg, exception=None ) :
        """Creates an exception. The message is required, but the exception
        is optional."""
        
        self._msg = msg
        self._exception = exception
        Exception.__init__( self, msg )

    def getMessage(self):
        """Return a message for this exception."""
        
        return self._msg

    def getException(self):
        """Return the embedded exception, or None if there was none."""
        
        return self._exception

    def __str__(self):
        """Create a string representation of the exception."""
        
        return self._msg

class NeuralWarning (NeuralException) :
   """Neural network warning exception. This is currently treated 
   exactly like an error and needs to be implemented properly."""
   
   pass

class NeuralParseException (NeuralException) :
   """Neural network parsing exception. Should be raised when there is
   any problem parsing either an AnnML or non-XML network file."""
   
   pass
