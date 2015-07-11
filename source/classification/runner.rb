# ooc, fei personal
# - vac

RFE=ARGV[1]

File.open(ARGV[0], 'w') do |file|
  file.puts "dataset,method,target,date,f1,accuracy"

  puts "per-subject"
  datasets = %w{au-counts koelstra-approach mahnob-hr-au-count mahnob-hr}
  datasets.each do |dataset|
    puts dataset
    (0..2).each do |target|
      score = `python cv_per_subject.py #{dataset} #{target} 3 #{RFE} | tail -n 1`
      puts score
      file.puts score
    end
  end

  puts "global"
  {
      # Mahnob
      'au-counts' => 3,
      'koelstra-approach' => 3,

      # Bined
      'bined' => 1,
      'bined-koelstra' => 1,

      # MIME
      'mime-au-counts' => 1,
      'mime-koelstra' => 1,

      # HR
      'mahnob-hr-au-count' => 3,
      'mahnob-hr' => 3,

      # Orientation
      'bined-orientation-au-counts' => 1,
      'bined-orientation' => 1,
      'mime-orientation-au-counts' => 1,
      'mime-orientation' => 1,
      'mahnob-orientation-au-counts' => 3,
      'mahnob-orientation' => 3,
  }.each do |dataset, truth_count|
    puts dataset
    score = `python cv_subjects.py #{dataset} 0 #{truth_count} #{RFE} | tail -n 1`
    puts score
    file.puts score
  end

  puts "Global training on mahnob"
  datasets = %w{mahnob-hr mahnob-hr-au-count au-counts}
  datasets.each do |dataset|
    (0..2).each do |target|
      score = `python cv_subjects.py #{dataset} #{target} 3 #{RFE} | tail -n 1`
      puts score
      file.puts score
    end
  end
end