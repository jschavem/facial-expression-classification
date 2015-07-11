require './init'

sessions = MahnobSession.load(2..4000)

Pipeline.new(
    :output_dir => 'mahnob-hr-au-count',
    :normalize => false,
    :flow => {[HeartRateProcessor] => [AverageAggregator, RmssdAggregator], [ActionUnitProcessor] => [AverageAggregator]}
).run(sessions)

