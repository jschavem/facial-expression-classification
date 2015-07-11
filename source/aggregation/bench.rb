require "rubygems"
require "stackprof"

require './init'

profile = StackProf.run(mode: :cpu) do
  require './mahnob_generator'
end
File.open('stackprof-cpu.dump', 'wb') { |f| f.write Marshal.dump(profile) }