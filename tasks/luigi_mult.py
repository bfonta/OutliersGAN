import os
import luigi
import mult

class Multiply(luigi.Task):
    path = luigi.Parameter()
    amax = luigi.IntParameter()
    bmax = luigi.IntParameter()

    def requires(self):
        return [
            MakeDirectory(path=os.path.dirname(self.path)),
        ]
                
    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        with open(self.path, 'w') as outf:
            for i in range(self.amax):
                for j in range(self.bmax):
                    result = mult.main(i,j)
                    outf.write(str(result)+'\n')
            outf.close()

class MakeDirectory(luigi.Task):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        os.makedirs(self.path)

if __name__ == '__main__':
   luigi.run()
