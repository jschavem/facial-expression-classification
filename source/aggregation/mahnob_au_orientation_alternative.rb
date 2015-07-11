require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-au+orientation',
    :normalize => false,
    :debug => true,
    :processors => [HeadOrientationProcessor, ActionUnitProcessor],
    :aggregator => AverageAggregator
).run(sessions)