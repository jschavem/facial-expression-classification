require './init'

sessions = BinedSession.load()

Pipeline.new(
    :output_dir => 'bined-orientation',
    :normalize => false,
    :debug => true,
    :flow => {[HeadOrientationProcessor] => [AverageAggregator]}
).run(sessions)