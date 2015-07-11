require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-orientation',
    :normalize => false,
    :debug => true,
    :flow => {[HeadOrientationProcessor] => [AverageAggregator]}
).run(sessions)