import subprocess 

PROGRAMS = [
            "kmeans_cpu"           ,
            "kmeans_cuda"          ,
            "kmeans_cuda_shared"   ,
           # "kmeans_thrust"
          ]

INPUTS = [
          "random-n2048-d16-c16.txt"    #,
         # "random-n16384-d24-c16.txt"  #,
         # "random-n65536-d32-c16.txt"
         ]

dimensions = {
              "random-n2048-d16-c16.txt"  : 16,
              "random-n16384-d24-c16.txt" : 24,
              "random-n65536-d32-c16.txt" : 32
             }

ITERATIONS = 1

for(program) in PROGRAMS:
    for filename in INPUTS:
        print("{}: {}".format(program, filename))
        for i in range(ITERATIONS):
            subprocess.call([
                "bin/{}".format(program), 
                "-i", "input/{}".format(filename), 
                "-k", str(16),
                "-d", str(dimensions[filename]),
                "-m", str(150),
                "-t", str(0.00001),
                "-c",   # comment out to print labels instead of centroids (c)
                "-s", str(8675309)])
        print("")
