using System;
using SimpleMath.Collections;

namespace SimpleML
{
    public interface ISLModel
    {
        public ISLModel Train(DoubleMatrix trainingdata, string[] labels, bool overwrite);

        public string Predict(double[] predictdata, out double[] output);

        public double Score(DoubleMatrix testdata, string[] labels);
    }
}
