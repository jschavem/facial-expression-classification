require './init'

sessions = MimeSession.load()

Pipeline.new(
    :output_dir => 'mime-orientation-au-counts',
    :normalize => false,
    :debug => false,
    :flow => {[HeadOrientationProcessor, ActionUnitProcessor] => [AverageAggregator]}
).run(sessions)