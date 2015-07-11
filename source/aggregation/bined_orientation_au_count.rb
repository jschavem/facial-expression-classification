require './init'

sessions = BinedSession.load()

Pipeline.new(
    :output_dir => 'bined-orientation-au-counts',
    :normalize => false,
    :debug => false,
    :flow => {[HeadOrientationProcessor, ActionUnitProcessor] => [AverageAggregator]}
).run(sessions)