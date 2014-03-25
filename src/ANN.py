#!/usr/bin/python
#-*-coding:utf-8-*-

import ANN_loader
import activation
import exceptions
import numpy as np   # change: horton -- there is no math.sqrt / or import math

import types
import random

class Neuron (object) :
   """Abstract neuron class. Parent of all other neuron classes."""
   
   def fire ( self ) :
      """Called when a neuron receives new input(s). The neuron 'activates'
      or transforms the data to ouput."""
      
      self.activate()

class InputNeuron ( Neuron, object ) :
   """Input neuron class. Input neurons are simple, they simply pass input as
   output"""
   
   def __init__ ( self ) :
      self._input = None
      self.output = None
      
   def activate ( self ) :
      """Pass input on as output."""
      
      if ( self.input is None ) :
         raise exceptions.NeuralException( exceptions.ERR_INPUTNONE )
      
      self.output = self.input

   def _geti ( self ) :
      """Return the input."""
   
      return self._input

   def _seti ( self, value ) :
      """Try to coerce whatever value is to a float."""
      
      try :
         self._input = float( value )

      except Exception, ex :
         raise exceptions.NeuralException( exceptions.ERR_INPUTTYPE, ex )

   input = property( _geti, _seti )					# ???

class HiddenNeuron (Neuron, object) :
   """Hidden neuron class. Hidden neurons transform each input given
   based on the weight associated with that input connection."""
   
   def __init__ ( self ) :
      self.weights = None
      self.weightdeltas = None
      self.inputsum = None

      # Default hidden node activation is sigmoidal. To change,
      # simply set neuron.activation to any of the activation
      # functions in the activation module after the neuron has been
      # instantiated.
      self.activation = activation.Sigmoid()
      
   def activate ( self ) :
      """Run the neuron's activation function. The function may be
      set to any of the classes provided in neural.activation or
      by a user supplied class."""
      
      if ( self.inputsum is None ) :
         raise exceptions.NeuralException( exceptions.ERR_INPUTNONE )

      if ( type( self.inputsum ) is not types.FloatType ) :
         raise exceptions.NeuralException( exceptions.ERR_INPUTTYPE )

      self.output = self.activation.activate( self.inputsum )		# Compute output
      
   def computedelta ( self, nextlayer, index ) : 
      """Compute the amount of error at this node for training so that 
      it can later be minimized.
      
      nextlayer -- the layer to which the outputs of this neuron are 
                   sent (the layer below the one this neuron is in)
      index -- the index in the current layer of this neuron"""

      if ( (nextlayer is None) ) : 
         raise exceptions.NeuralException( exceptions.ERR_LAYERNONE )

      if ( (index < 0) or (type( index ) != types.IntType) ) :
         raise exceptions.NeuralException( exceptions.ERR_INDEX )
      
      sum = 0.0

      for neuron in nextlayer.neurons :				# sum = Ek(Wkj*delt-k)
         try :
            sum += neuron.weights[index] * neuron.delta
         except Exception, ex :
            raise exceptions.NeuralException( exceptions.ERR_INDEX, ex )

      self.delta = self.output * (1 - self.output) * sum		# delt-j = sum * Hj(1-Hj)

   def updateweight ( self, learningrate, momentum ) :
      """Update the weights for each input connection. 
      
      learningrate -- learning rate requested by the network		
      momentum -- momentum requested by the network"""			# the rate of the old weight in new weight

      # Try to coerce the learning rate and momentum to a float
      try : 
         learningrate = float( learningrate )
         momentum = float( momentum )

      except Exception, ex :
         raise exceptions.NeuralException( exceptions.ERR_PARAMTYPE, ex )

      if ( type( learningrate ) != types.FloatType ) :
         raise exceptions.NeuralException( exception.ERR_PARAMTYPE )

      if ( type( momentum ) != types.FloatType ) :
         raise exceptions.NeuralException( exceptions.ERR_PARAMTYPE )

      if ( self.delta is None ) :
         raise exceptions.NeuralException( exceptions.ERR_DELTANONE )
      
      # Update each weight for this neuron.
      for i in xrange( len( self.weights ) ) :				# Wi = m*Wi + n*Xi*delt-j
         self.weightdeltas[i] = ( learningrate * self.input[i] * self.delta ) + ( momentum * self.weightdeltas[i] )
	 self.weights[i] += self.weightdeltas[i]
      
   def _seti ( self, value ) :
      """Set input vector, and compute the input sum that will be
      used by the transfer function to compute this neuron's output.
      The first time through the input vector is assumed to be the 
      correct size, and the weight vector is initialized to be the
      same size as the input vector with random values for the weights.
      Each time after the initial call, the weight and input vector 
      size is assumed fixed.
      
      value -- object on the RHS of the `=' operator"""
      
      if ( types.ListType is not type( value ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )

      if ( 0 == len( value ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      # On the first assignment of input, set the weights to a random
      # initial value. Wait until we receive an input vector, because
      # otherwise there's no way of knowing how many weights are needed.
      # Trust the input vector to have the correct number. This is a 
      # potential bug.
      if ( self.weights is None ) :
         self.weightdeltas = [0.0] * len( value )
         self.weights = []
         for i in value :
            self.weights.append( random.uniform(-1, 1) )		# Wi started with a random between -1 and 1

      # Make sure the input vector and the weights are the same size
      # vectors
      if ( len( value ) != len( self.weights ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
      
      # Compute the input sum to the transfer function.
      self.inputsum = 0.0
      for i in xrange( len( self.weights ) ) :				# Calculate the output by input and weight
         try :
            self.inputsum += ( float( value[i] ) * self.weights[i])

         except Exception, ex :
            raise exceptions.NeuralException( exceptions.ERR_INDEX, ex )
      
      self._input = value
      
   def _geti ( self ) : 
      """Get input vector."""
      
      return self._input
      
   input = property( _geti, _seti )							# ???
   
class OutputNeuron ( HiddenNeuron, object ) :
   """Output neuron class. Exactly the same as a hidden neuron, except that
   the neuron's delta is computed slightly differently."""
   
   def computedelta ( self, expected ) :
      """Compute the amount of error at this node for training so that it can
      later be minimized."""
      
      self.delta = self.output * ( 1.0 - self.output ) * ( expected - self.output )	# delt-k = Yk(1-Yk)(Tk-Yk)

class Layer (object) :
   """Abstract layer class. All layers are derived from this class."""
   
   def feedforward ( self ) :
      """Push input through the layer. Input must have been set prior to
      calling."""
      
      for i in xrange( len( self.neurons ) ) :
         self.neurons[i].fire()

   def _geto ( self ) :
      """Get output vector."""
      
      self._output = []						# all output of neurons in this layers
      for n in self.neurons :
         self._output.append( n.output )

      return self._output
	 
   output = property( _geto )
      
class InputLayer ( Layer, object ) :		 	# the input and output is same in this layers
   """Input layer class. Receives an input vector as input and passes it on as
   output."""
   
   def __init__ ( self, numNeurons ) :
      self.neurons = []
      for i in xrange( numNeurons ) :
         self.neurons.append( InputNeuron() )			# each element is a InputNeron object

   def _seti ( self, value ) :
      """Set input vector."""
      
      if ( types.ListType is not type( value ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      if ( len( value ) != len( self.neurons ) ) :		# length of value List must == length of neurons
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )

      for i in xrange( len( self.neurons ) ) :
         if ( types.FloatType is not type( value[i] ) ) :
            raise exceptions.NeuralException( exceptions.ERR_INPUTTPE )
	 
	 self.neurons[i].input = value[i]

      self._input = value

   def _geti ( self ) :
      """Get input vector."""
      
      return self._input

   input = property( _geti, _seti )
      
class HiddenLayer ( Layer, object ) :
   """Hidden layer class. Hidden layers 'learn' an internal representation 
   of the input data by separating it into non-linear hyperplanes. A 
   network can have an arbitrary number of hidden layers, but it's been 
   proven that a network with two hidden layers can learn any pattern."""
   
   def __init__ ( self, numNeurons ) :
      self.neurons = []
      for i in xrange( numNeurons ) :
         self.neurons.append( HiddenNeuron() )			# each element is a HiddenNeuron object

   def backpropagate ( self, learningrate, momentum, nextlayer ) :
      """Propagate the error from the next layer up through this one and to 
      the previous one."""
      
      for i in xrange( len( self.neurons ) ) :
         self.neurons[i].computedelta( nextlayer, i )		# Propagate new error delta from the nextlayer
	 self.neurons[i].updateweight( learningrate, momentum )
      
   def _seti ( self, value ) :					# Set the input value of each neuron in this layer
      """Set input vector."""
      
      if ( types.ListType is not type( value ) ) :	
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )

      for i in xrange( len( self.neurons ) ) :
         self.neurons[i].input = value

      self._input = value

   def _geti ( self ) :
      """Get input vector."""
      
      return self._input

   input = property( _geti, _seti )

class ContextLayer ( HiddenLayer, object ) :
   pass

class OutputLayer ( HiddenLayer, object ) :
   """Output layer class. Output layers 'learn' an internal representation of
   a network the same way that hidden layers. However, instead of 
   propagating the error from the previous layer, the error from the 
   actual output is propagated upwards."""
   
   def __init__ ( self, numNeurons ) :
      self.neurons = []
      for i in xrange( numNeurons ) :
         self.neurons.append( OutputNeuron() )			# each element is a OutputNeuron object

   def backpropagate ( self, learningrate, momentum, expected ) :
      """Propagate the error from the output up through this layer and to 
      the previous one.
      
      learningrate -- learning rate to be applied to weight changes
      momentum -- momentum with which the weights change
      expected -- expected output vector"""
	    
      if ( types.ListType is not type( expected ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
      
      for i in xrange( len( self.neurons ) ) :
         self.neurons[i].computedelta( expected[i] )		# OutputNeuron Compute the delta by expected values and self.output
	 self.neurons[i].updateweight( learningrate, momentum )
      
class BackPropNet ( object ) :
   """Backpropagation neural network class. Supports an arbitrary number of
   hidden layers."""
   
   def __init__ ( self, inputs = 0, outputs = 0 ) :
      """inputs -- number of inputs to the network
      outputs -- number of outputs generated by the network"""
      
      self.hiddenlayers = []					# There maybe several layer of Hiddenlayers
      self.inputlayer = InputLayer( inputs )
      self.outputlayer = OutputLayer( outputs )			# There should be only one layser of inputlayer and outputlayer
      
      self.sumsqerr = 0.0
      self.rmserr = 0.0
      self.type = 'bpn'

   def addhidden ( self, nodes ) :				# each time add one layer
      """Add a hidden layer to the network.

      nodes -- number of neurons in the layer to create"""
      
      self.hiddenlayers.append( HiddenLayer( nodes ) )

   def addinput ( self, nodes ) :
      """Add an input layer to the network. Any existing input layer
      will be overwritten.

      nodes -- number of neurons in the layer to create"""

      self.inputlayer = InputLayer( nodes )

   def addouput ( self, nodes ) :
      """Add an output layer to the network. Any existing output layer
      will be overwritten.

      nodes -- number of neurons in the layer to create"""

      self.outputlayer = OutputLayer( nodes )

   def _feedforward ( self, inputs ) :				# from input to output
      """Feed one input vector through the network and generate an ouput
      vector.
      
      inputs -- input vector"""

      if ( types.ListType is not type( inputs ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      if ( len( inputs ) != len( self.inputlayer.neurons ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      if ( 0 == len( self.hiddenlayers ) ) :
         # NOTE : In the current implementation this is actually an error
         # TODO : It should be fine to allow someone to create a simple perceptron
         #        How to deal with this?
         raise exceptions.NeuralWarning( exceptions.ERR_LAYERNOHID )

      # Feed through the input layer
      self.inputlayer.input = inputs
      self.inputlayer.feedforward()

      # Feed through the hidden layers
      for i in xrange( len( self.hiddenlayers ) ) :		# feedforward() is a method of the abstract class--layer : each neuron.fire()
         if ( i == 0 ) :
	    self.hiddenlayers[i].input = self.inputlayer.output
	 else :
	    self.hiddenlayers[i].input = self.hiddenlayers[i-1].output

	 self.hiddenlayers[i].feedforward()

      # And feed through the output layer
      self.outputlayer.input = self.hiddenlayers[len( self.hiddenlayers ) - 1].output
      self.outputlayer.feedforward()

      self.output = self.outputlayer.output

   def _backpropagate ( self, expected, learningrate, momentum ) :
      """Propagate the error of the current output compared with the expected
      output through the network. 
      
      expected -- expected output vector
      learningrate -- learning rate to be applied to the weight changes
      momentum -- momentum with which the weights change"""
      
      if ( types.ListType is not type( expected ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      if ( len( expected ) != len( self.outputlayer.neurons ) ) :
         raise exceptions.NeuralException( exceptions.ERR_VECTOR )
         
      if ( 0 == len( self.hiddenlayers ) ) :
         # NOTE : In the current implementation this is actually an error
         # TODO : It should be fine to allow someone to create a simple perceptron
         #        How to deal with this?
         raise exceptions.NeuralWarning( 'Network has no hidden layers.' )

      self.outputlayer.backpropagate( learningrate, momentum, expected )

      self.hiddenlayers.reverse()				# from last to first
      
      for i in xrange( len( self.hiddenlayers ) ) :
         if ( i == 0 ) :
	    self.hiddenlayers[i].backpropagate( learningrate, momentum, self.outputlayer ) 
	 else :
	    self.hiddenlayers[i].backpropagate( learningrate, momentum, self.hiddenlayers[i-1] )
	    
      self.hiddenlayers.reverse()				# reveser back !

   def train ( self, exemplar, learningrate = 0.5, momentum = 0.9 ) :		# examplar: ([input List],[output List])
      """Train the network on one exemplar. An exemplar is a training
      set pair which consists of ([input vector], [expected output vector])
      
      exemplar -- exemplar to train
      learningrate -- learning rate to be applied to the weight changes
      momentum -- momentum with which the weights change"""
      
      if ( types.TupleType is not type( exemplar ) ) :
         raise exceptions.NeuralException( exceptions.ERR_PARAMTYPE )
         
      if ( 0 == len( exemplar ) ) :
         raise exceptions.NeuralException( exceptions.ERR_PARAMVAL  )
         
      if ( 0 == len( self.hiddenlayers ) ) :
         # NOTE : In the current implementation this is actually an error
         # TODO : It should be fine to allow someone to create a simple perceptron
         #        How to deal with this?
         raise exceptions.NeuralWarning( exceptions.ERR_LAYERNOHID )

      self._feedforward( exemplar[0] )				# set input for each layer
      self._backpropagate( exemplar[1], learningrate, momentum )# reset weight for each layer

   def learn ( self, exemplars, epochs = 1, learningrate = 0.5, momentum = 0.9 ) :
      """Teach the network to recognize a set of exemplars by training the
      entire set a number of times.
      
      exemplars -- a list containing exemplar training pairs
      epochs -- the number of epochs to teach : ages / times
      learningrate -- learning rate to be applied to the weight changes
      momentum -- momentum with which the weights change"""
      
      if ( types.ListType is not type( exemplars ) ) :			# examplars: [([input List],[output List]),([input List],[output List])... ]
         raise exceptions.NeuralException( exceptions.ERR_PARAMTYPE )
      
      # Train each exemplar 'epochs' times
      for i in xrange( epochs ) :
         self.sumsqerr = 0.0
	 
         for ex in exemplars :
            self.train( ex, learningrate, momentum ) 

            # Update the sum squared error
            for j in xrange( len( self.outputlayer.neurons ) ) :
	       self.sumsqerr = ( ex[1][j] - self.outputlayer.neurons[j].output ) * ( ex[1][j] - self.outputlayer.neurons[j].output )

	 # Compute the root mean square error for the current epoch.
	 self.rmserr = np.sqrt( self.sumsqerr / (len( exemplars ) * len( self.outputlayer.output ) ) )

   def run ( self, inputs ) :
      """Run a set of input vectors through the (trained) network. Returns
      the output vectors produced.
      
      inputs -- set of input vectors"""
      
      resultvector = []

      for i in inputs :
         self._feedforward( i )
	 resultvector.append( self.output )

      return resultvector
	 
   def save ( self, filename ) :
      """Save the current network configuration to a file.
      NOTE: Deprecated, use xsave and xload

      filename -- name of the file to save"""
      
      outfile = open( filename, 'w' )
      outfile.write( 'i:' + str( len( self.inputlayer.neurons ) ) + "\n" )
      
      for i in xrange( len( self.hiddenlayers ) ) :
         # For each hidden layer, write the number of neurons followed by 
	 # a line with the weight vectors of each neuron in the layer.
	 # 
	 
         outfile.write( 'h:' + str( len( self.hiddenlayers[i].neurons ) ) + "\n" )
	 
	 nstr = ''
	 for neuron in self.hiddenlayers[i].neurons :			# each layer has several neurons, each neuron has several weights
	    wstr = ''
	    for weight in neuron.weights :
	       wstr += str( weight ) + ','
	       
	    nstr += wstr.rstrip(',') + ':'

	 outfile.write( nstr.rstrip(':') + "\n" )			#   i:3		-- 3 input neurons
									#   h:2		-- this hiddenlayer got 2 neurons
									#   0.1,0.3,0.3:0.2,0.3,0.5
									#   h:2
									#   0.3,0.4:0.5,0.1
									#   o:2		-- 2 output neurons
									#   0.2,0.7:0.9,0.3


      outfile.write( 'o:' + str( len( self.outputlayer.neurons ) ) + "\n" )
      nstr = ''
      for neuron in self.outputlayer.neurons :
         # Write the weight vectors for each neuron in the output layer.
         wstr = ''
	 for weight in neuron.weights :
	    wstr += str( weight ) + ','

	 nstr += wstr.rstrip(',') + ':'

      outfile.write( nstr.rstrip(':') + "\n" )
      outfile.close()
      
   def load ( self, filename ) :
      """Load a network configuration from a file.
      NOTE: Deprecated, use xsave and xload
      
      filename -- name of the file to load"""
   
      # Clear out the current network, if there is one
      self.inputlayer = None
      self.outputlayer = None
      self.hiddenlayers = []
      
      infile = open( filename, 'r' )

      # Load the input layer.
      line = infile.readline()
      tokens = line.split( ':' )
      
      if ( tokens[0] != 'i' ) :
         raise exceptions.NeuralParseException( 'Expected input layer token, encountered: ' + tokens[0] + '.' )

      self.inputlayer = InputLayer( int( tokens[1] ) )

      while ( infile ) :				# !!! if there is no "o" in this file, it will never end
         # Loop through the hidden layers.
	 
         line = infile.readline()
	 tokens = line.split( ':' )
	 
	 # Finished with the hidden layers if read an output layer.
	 if ( tokens[0] == 'o' ) : break 

         if ( tokens[0] != 'h' ) :
            raise exceptions.NeuralParseException( 'Expected hidden or output layer token, encountered: ' + tokens[0] + '.' )

	 self.addhidden( int( tokens[1] ) )

	 # Read the weights for the newly created layer.
	 line = infile.readline()
	 tokens = line.split( ':' )
	 
	 for i in xrange( len( tokens ) ) :
	    tokens[i] = tokens[i].split( ',' )

	    for j in xrange( len ( tokens[i] ) ) :
	       tokens[i][j] = float( tokens[i][j] )

	 for i in xrange( len( self.hiddenlayers[-1].neurons ) ) :		# ?: [-1] : the new added hiddenlayer is the last layer of all
	    self.hiddenlayers[-1].neurons[i].weights = tokens[i][:]		# copy tokens[i] but not point to

      # Load the output layer.
      tokens = line.split( ':' )
     
      self.outputlayer = OutputLayer( int( tokens[1] ) )
     
      line = infile.readline()
      tokens = line.split( ':' )
     
      for i in xrange( len( tokens ) ) :
	 tokens[i] = tokens[i].split( ',' )

	 for j in xrange( len ( tokens[i] ) ) :
            tokens[i][j] = float( tokens[i][j] )

         for i in xrange( len( self.outputlayer.neurons ) ) :
	    self.outputlayer.neurons[i].weights = tokens[i][:]

      infile.close()

   def xsave ( self, filename ) :
      """Save the network to an AnnML XML file. This should be used
      instead of class.save().
      
      filename -- name of the file to save"""
      
      ANN_loader.XMLNeuralSaver().save( self, filename )
      
   def xload ( self, filename ) :
      """Load a network from an AnnML XML file. This should be used
      instead of class.load().
      
      filename -- name of the file to load"""
      
      tempnetwork = ANN_loader.XMLNeuralLoader().load( filename )

      if ( BackPropNet != type( tempnetwork ) ) :
         raise exceptions.NeuralException( exceptions.ERR_NETWORKTYPE )
      else :
         self.inputlayer = tempnetwork.inputlayer
         self.hiddenlayers = tempnetwork.hiddenlayers
         self.outputlayer = tempnetwork.outputlayer

class ElmanNetwork (BackPropNet, object) :
   def __init__ ( self, inputs = 0, outputs = 0 ) :
      BackPropNet.__init__( inputs, outputs )

      self.type = 'elman'
   
