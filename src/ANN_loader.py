#!/usr/bin/python
#-*-coding:utf-8-*-

import ANN
from xml.sax._exceptions import *
from xml.sax import make_parser
from xml.sax.handler import ContentHandler
from xml.sax import saxutils

class XMLBPNSaver :
   """Saves a Backpropagation Network to an AnnML file."""
   
   def _writeweight ( self, value, outfile ) :
      """Write a weight in XML.
      
      value -- the weight value as a float
      outfile -- file object opened for writing"""
      
      print >> outfile, '<weight>' + str( value ) + '</weight>'
   
   def _writeneuron ( self, neuron, outfile ) :
      """Write a neuron in XML.
      
      neuron -- neuron object to write
      outfile -- file object opened for writing"""
      
      if ( ANN.InputNeuron == type( neuron ) ) :
         print >> outfile, '<neuron />'		# there is no weights of InputNeuron
      else :
         print >> outfile, '<neuron>'			# BEGIN
         
         for w in neuron.weights :
            self._writeweight( w, outfile )
            
         print >> outfile, '</neuron>'			# END

	# miss the exception treating
      
   def _writelayer ( self, layer, outfile ) :
      """Write a layer in XML.
      
      layer -- layer object to write
      outfile -- file object opened for writing"""

      if (ANN.InputLayer == type( layer ) ) :
         print >> outfile, '<layer type="input" neurons="' + str( len( layer.neurons ) ) + '">'
      elif ( ANN.HiddenLayer == type( layer ) ) :
         print >> outfile, '<layer type="hidden" neurons="' + str( len( layer.neurons ) ) + '">'
      elif ( ANN.OutputLayer == type( layer ) ) :
         print >> outfile, '<layer type="output" neurons="' + str( len( layer.neurons ) ) + '">'
      else:
         raise Exception( 'Invalid layer type.' )
         
      for n in layer.neurons : 
         self._writeneuron( n, outfile )            

      print >> outfile, '</layer>'

   def save ( self, network, xmlfile ) :
      """Save a backpropagation network to an AnnML file.

      network -- the network object to save
      xmlfile -- the filename to save to"""
      
      outfile = open( xmlfile, 'w' )

      print >> outfile, '<?xml version="1.0"?>'
      print >> outfile, '<network type="bpn">'

      self._writelayer( network.inputlayer, outfile )
      for l in network.hiddenlayers :
         self._writelayer( l, outfile )
      self._writelayer( network.outputlayer, outfile )

      print >> outfile, '</network>'

      outfile.close()

class XMLNeuralSaver :
   """XML Saver wrapper. Instantiates the correct network type class
   for writing that network's output."""
      
   def save ( self, network, xmlfile ) :
      """Save the network to a file.
      
      network -- the network object to save
      xmlfile -- the filename to save to"""
      
      if ( ANN.BackPropNet == type( network ) ) :
         XMLBPNSaver().save( network, xmlfile )			# Create a XMLBPNSaver object
      else :
         raise Exception( 'Unsupported network type.' )

class XMLNeuralLoader :
   """Neural nework loader class. Loads any network that has been 
   implemented."""
   
   def load ( self, xmlfile ) :
      """Constructs a neural network described by an AnnML XML file.
      Returns the resulting network."""
      
      xmlch = XMLNeuralHandler()
      parser = make_parser()
      parser.setContentHandler( xmlch )				# Configurate the self-defined ContentHandler

      try : 
         parser.parse( xmlfile )
         return xmlch.getNetwork()

      except SAXParseException, ex :
         raise Exception( ex.__str__() )

class XMLNeuralHandler (ContentHandler) :			# Realize the metod of startElemnt( ) and endElement( ) : basic for ContentHandler
   """Neural network content handler. Contructs the layers and
   network described in an AnnML XML file."""
   
   _inweight = False
   _currentlayer = None
   _currentneuron = 0
   _currentweight = 0

   _weights = []

   _inputlayer = None
   _hiddenlayers = []
   _outputlayer = None
   
   def startElement ( self, name, attrs ) :
      if ( 'network' == name ) :
         self._createNet( attrs.get( 'type' ) )
         self._currentlayer = None			# reset for another XMLNeural loading , in the case of no network endElement
      elif ( 'layer' == name ) : 
         self._addLayer( attrs.get( 'type' ), attrs.get( 'neurons' ) )
         self._currentneuron = 0			# reset for another layer loading, in the case of no layer endElement
      elif ( 'neuron' == name ) :
         self._currentweight = 0			# reset for another neuron loading, in the case of no neuron endElement
      elif ( 'weight' == name ) :
         self._inweight = True
      else :
         raise Exception( 'Invalid/Unknown markup tag : ' + name ) 

   def endElement ( self, name ) :
      if ( 'neuron' == name ) :
         if ( ANN.InputLayer != type( self._currentlayer ) ) :
            self._currentlayer.neurons[self._currentneuron].weights = self._weights	# add  
            self._weights = []				# !!! The left has already been initialized, so it's not a point to the right !!!
         
         self._currentneuron += 1
      elif ( 'weight' == name ) :
         self._currentweight += 1
         self._inweight = False
      elif ( 'layer' == name ) :
         self._currentlayer = None

   def characters ( self, ch ) :
      if ( self._inweight ) :
         self._weights.append( float( ch ) )

   def _createNet ( self, type ) :
      """Instantiate the correct network type attribute from the
      XML file. Complain if there is no support for the specified
      type."""

      if ( 'bpn' == type ) :
         self._network = ANN.BackPropNet()
         self._nettype = 'bpn'
      else :
         raise Exception( 'Unsupported network type : ' + type )

   def _addLayer ( self, type, neurons ) :		 # add empty neurons
      """Add a layer to the network of a specified type. Complain
      if there is no support for the specified type."""
      
      if ( 'input' == type ) :
         self._currentlayer = self._inputlayer = ANN.InputLayer( int( neurons ) )
      elif ( 'hidden' == type ) :
         self._currentlayer = ANN.HiddenLayer( int( neurons ) )
         self._hiddenlayers.append( self._currentlayer )
      elif ( 'output' == type ) :
         self._currentlayer = self._outputlayer = ANN.OutputLayer( int( neurons ) )
      else :
         raise Exception( 'Unsupported layer type : ' + type )

   def getNetwork ( self ) :
      """Construct a network from the layers created after parsing and
      return the network."""
      
      if ( 'bpn' == self._nettype ) :
         if ( (self._inputlayer is None) or (self._outputlayer is None) ) :
            raise Exception( 'Parse error, no input or output layer' )
         else :    
            self._network.inputlayer = self._inputlayer
            self._network.hiddenlayers = self._hiddenlayers
            self._network.outputlayer = self._outputlayer
      else :
         raise Exception( 'Unsupported network type : ' + type )

      return self._network

