require './init'

sessions = MimeSession.load()

Pipeline.new(
    :output_dir => 'mime-orientation',
    :normalize => false,
    :debug => false,
    :flow => {[HeadOrientationProcessor] => [AverageAggregator]}
).run(sessions)