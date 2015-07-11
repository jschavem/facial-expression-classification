require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'au-counts-hr',
    :normalize => false,
    :debug => true,
    :processors => [ActionUnitProcessor, HeartRateProcessor],
    :aggregator => AverageAggregator
).run(sessions)