ROOT_DIR = File.dirname(__FILE__)
DATA_DIR = File.join(ROOT_DIR, '..', 'data')

require 'rubygems'
require 'bundler/setup'

require 'pp'

#$:.unshift File.join(File.dirname(__FILE__), 'lib')
#$:.unshift File.join(File.dirname(__FILE__), 'lib', 'filters')

require './lib/mahnob_conversion'
require './lib/bined_conversion'
require './lib/face_reader_conversion'
require './lib/mime_conversion'
require './lib/emfacs'

require '../data/stimulus_lengths'

require './lib/session'
require './lib/mahnob_session'
require './lib/bined_session'
require './lib/mime_session'
require './lib/face_reader_session'
require './lib/metadata_file'
require './lib/example_session'

require './lib/processors/dominant_state_processor'
require './lib/processors/dominant_emotion_processor'
require './lib/processors/action_unit_processor'
require './lib/processors/emfacs_action_unit_processor'
require './lib/processors/head_orientation_processor'
require './lib/processors/heart_rate_processor'

require './lib/aggregators/average_aggregator'
require './lib/aggregators/weighted_aggregator'
require './lib/aggregators/rmssd_aggregator'

require './lib/pipeline'