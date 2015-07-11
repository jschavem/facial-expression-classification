require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-orientation-au-counts',
    :normalize => false,
    :debug => false,
    :flow => {[HeadOrientationProcessor, ActionUnitProcessor] => [AverageAggregator]}
).run(sessions)