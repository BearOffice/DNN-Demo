using System;
using System.Collections.Generic;
using System.Linq;
using SimpleMath;
using SimpleMath.Collections;
using SimpleMath.MathQ;
using SimpleMath.Supports;
using SimpleML.Support;

namespace SimpleML
{
    public class NeuralNetwork : ISLModel
    {
        private readonly int[] _hiddenLayerSizes;
        private readonly int _inputNum;
        private readonly string[] _outputLabels;
        private readonly int _learningRounds;
        private readonly int _miniBatchSize;
        private readonly double _learningRate;
        private readonly int _randomSeed;
        private List<DoubleMatrix> _layerWeights;
        private List<DoubleMatrix> _layerBiases;  // 1-D matrix

        public string[] OutPutLabel { get => _outputLabels.ToArray(); }
        public DoubleMatrix[] LayerWeights { get => _layerWeights.ToArray(); }
        public DoubleMatrix[] LayerBiases { get => _layerBiases.ToArray(); }

        public NeuralNetwork(int inputnum, int[] hiddenlayersizes, string[] outputlabels, 
            int learningrounds, int minibatchsize, double learningrate, int? randomseed = null)
        {
            _inputNum = inputnum;
            _hiddenLayerSizes = hiddenlayersizes;
            _outputLabels = outputlabels;
            _learningRounds = learningrounds;
            _miniBatchSize = minibatchsize;
            _learningRate = learningrate;
            _randomSeed = randomseed ?? new Random().Next();
        }

        public ISLModel Train(DoubleMatrix trainingdata, string[] labels, bool overwrite = false)
        {
            if (trainingdata.ColumnsNum != _inputNum || trainingdata.RowsNum != labels.Length)
                throw new ArgumentException("Training data are inconsistent.");

            if (overwrite || _layerWeights == null || _layerWeights == null)
                Initialize(_inputNum, _outputLabels.Length);

            for (var i = 0; i < _learningRounds; i++)
            {
                GenRandomMiniBatches(trainingdata, labels)
                     .ForEach(minipatches => UpdateNetwork(minipatches.Item1, minipatches.Item2));
            }
            
            return this;

            List<(DoubleMatrix, string[])> GenRandomMiniBatches(DoubleMatrix matrix, string[] labels)
            {
                var indexlist = Enumerable.Range(0, matrix.RowsNum);
                var shuffledlist = indexlist.OrderBy(a => Guid.NewGuid()).ToList();

                var splittedlist = new List<List<int>>();


                for (var i = 0; i < matrix.RowsNum; i += _miniBatchSize)
                {
                    if (i + _miniBatchSize < matrix.RowsNum)
                        splittedlist.Add(shuffledlist.GetRange(i, _miniBatchSize));
                    else
                        splittedlist.Add(shuffledlist.GetRange(i, matrix.RowsNum - i));
                }

                return splittedlist.Select(sfindexlist =>
                {
                    var sfmatrix = sfindexlist
                        .Select(sfindex => matrix.GetRow(sfindex).ToDoubleMatrix())
                        .Aggregate((acc, next) => acc | next);

                    var sflabels = sfindexlist.Select(sfindex => labels[sfindex]).ToArray();

                    return (sfmatrix, sflabels);
                }).ToList();
            }
        }

        private void Initialize(int innum, int outnum)
        {
            // layer weights: 1 + (hidden - 1) + 1
            _layerWeights = new List<DoubleMatrix>();

            _layerWeights.Add(GenRandomMatrix(_hiddenLayerSizes[0], innum));
            for (var i = 0; i < _hiddenLayerSizes.Length - 1; i++)
            {
                _layerWeights.Add(GenRandomMatrix(_hiddenLayerSizes[i + 1], _hiddenLayerSizes[i]));
            }
            _layerWeights.Add(GenRandomMatrix(outnum, _hiddenLayerSizes[^1]));


            // layer biases: hidden + 1
            _layerBiases = new List<DoubleMatrix>();

            for (var i = 0; i < _hiddenLayerSizes.Length; i++)
            {
                _layerBiases.Add(new DoubleMatrix(_hiddenLayerSizes[i], 1));
            }
            _layerBiases.Add(new DoubleMatrix(outnum, 1));


            DoubleMatrix GenRandomMatrix(int rows, int columns)
            {
                var randnd = new RandomND(_randomSeed, 0, Math.Sqrt(2.0 / columns));

                return new DoubleMatrix(rows, columns)
                    .Set((_, _) => randnd.Next()).ToDoubleMatrix();
            }
        }

        private void UpdateNetwork(DoubleMatrix trainingdata, string[] labels)
        {
            var delweights = new List<DoubleMatrix>();
            _layerWeights.ForEach(weight => delweights.Add(
                new DoubleMatrix(weight.RowsNum, weight.ColumnsNum)));

            var delbiases = new List<DoubleMatrix>();
            _layerBiases.ForEach(bias => delbiases.Add(
                new DoubleMatrix(bias.RowsNum, bias.ColumnsNum)));

            for (var i = 0; i< labels.Length; i++)
            {
                var singledata = trainingdata.GetRow(i).Transpose().ToDoubleMatrix();
                var label = labels[i];

                var (deltadelweights, deltadelbiases) = BackPropagation(singledata, label);

                delweights = delweights.Zip(deltadelweights)
                    .Select(item => item.First + item.Second)
                    .ToList();

                delbiases = delbiases.Zip(deltadelbiases)
                    .Select(item => item.First + item.Second)
                    .ToList();
            }
            
            _layerWeights = _layerWeights.Zip(delweights)
                .Select(item => item.First - (_learningRate / _miniBatchSize) * item.Second)
                .ToList();

            _layerBiases = _layerBiases.Zip(delbiases)
                .Select(item => item.First - (_learningRate / _miniBatchSize) * item.Second)
                .ToList();
        }

        private DoubleMatrix ComputeOutputLayer(DoubleMatrix inputlayer,
            out List<DoubleMatrix> activationlist, out List<DoubleMatrix> zlist)
        {
            activationlist = new List<DoubleMatrix>();
            zlist = new List<DoubleMatrix>();

            activationlist.Add(inputlayer);

            // hidden layer
            for (var i = 0; i < _layerWeights.Count - 1; i++)
            {
                var prevactivation = activationlist[^1];
                var weight = _layerWeights[i];
                var bias = _layerBiases[i];
                
                var z = weight * prevactivation + bias;
                var activation = ReLU(z);

                zlist.Add(z);
                activationlist.Add(activation);
            }

            // output layer
            var lprevactivation = activationlist[^1];
            var lweight = _layerWeights[^1];
            var lbias = _layerBiases[^1];

            var lz = lweight * lprevactivation + lbias;
            var lactivation = SoftMax(lz);

            zlist.Add(lz);
            activationlist.Add(lactivation);


            return activationlist[^1];
        }

        private (List<DoubleMatrix>, List<DoubleMatrix>) BackPropagation(
            DoubleMatrix singledata, string label)
        {
            var delweights = new List<DoubleMatrix>();
            _layerWeights.ForEach(weight => delweights.Add(
                new DoubleMatrix(weight.RowsNum, weight.ColumnsNum)));

            var delbiases = new List<DoubleMatrix>();
            _layerBiases.ForEach(bias => delbiases.Add(
                new DoubleMatrix(bias.RowsNum, bias.ColumnsNum)));

            _ = ComputeOutputLayer(singledata, out var activationlist, out var zlist);

            var delta = PointwiseProduct(DerivsOfCost(activationlist[^1], label), 
                DerivsOfReLU(zlist[^1]));
            delweights[^1] = delta * activationlist[^2].Transpose().ToDoubleMatrix();
            delbiases[^1] = delta;

            for (var i = 2; i < activationlist.Count; i++)
            {
                delta = PointwiseProduct(_layerWeights[^(i - 1)].Transpose().ToDoubleMatrix() * delta,
                    DerivsOfReLU(zlist[^i]));
                delweights[^i] = delta * activationlist[^(i + 1)].Transpose().ToDoubleMatrix();
                delbiases[^i] = delta;
            }

            return (delweights, delbiases);
        }

        private static DoubleMatrix ReLU(DoubleMatrix matrix)
            => matrix.Map(num => num > 0 ? num : 0.0).ToDoubleMatrix();

        private static DoubleMatrix DerivsOfReLU(DoubleMatrix matrix)
            => matrix.Map(num => num > 0 ? 1.0 : 0.0).ToDoubleMatrix();

        private static DoubleMatrix SoftMax(DoubleMatrix matrix)
        {
            var numsexp = matrix.Map(num => Math.Exp(num)).ToDoubleMatrix();

            var numssum = 0.0;
            numsexp.Iterate(num => numssum += num);

            return numsexp.Map(num => num / numssum).ToDoubleMatrix();
        }

        private DoubleMatrix DerivsOfCost(DoubleMatrix outputlayer, string label)
        {
            var desiredoutput = new DoubleMatrix(outputlayer.RowsNum, outputlayer.ColumnsNum);
            desiredoutput[_outputLabels.ToList().IndexOf(label), 0] = 1.0; // columns = 0, 1-D matrix

            return outputlayer - desiredoutput;
        }

        private static DoubleMatrix PointwiseProduct(DoubleMatrix left, DoubleMatrix right)
            => left.Zip(right).Map(item => item.Item1 * item.Item2).ToDoubleMatrix();
        
        public string Predict(double[] predictdata, out double[] output)
        {
            var inputlayer = new DoubleMatrix(predictdata).Transpose().ToDoubleMatrix();
            var outputlayer = ComputeOutputLayer(inputlayer, out _, out _);
            var predictlabel = _outputLabels[
                GetMaxPosition(Matrix.Convert1DMatrixToArray(outputlayer))];

            output = Matrix.Convert1DMatrixToArray(outputlayer);
            return predictlabel;


            static int GetMaxPosition(double[] array)
            {
                var pos = 0;
                var max = 0.0;

                for (var i = 0; i < array.Length; i++)
                {
                    if (array[i] > max)
                    {
                        max = array[i];
                        pos = i;
                    }
                }

                return pos;
            }
        }

        public double Score(DoubleMatrix testdata, string[] labels)
        {
            if (testdata.RowsNum != labels.Length)
                throw new ArgumentException("Test data are inconsistent.");

            var correct = 0;

            for (var i = 0; i < testdata.RowsNum; i++)
            {
                var inputlayer = Matrix.Convert1DMatrixToArray(
                    testdata.GetRow(i).Transpose().ToDoubleMatrix());

                if (Predict(inputlayer, out _) == labels[i]) correct++;
            }

            return (double)correct / testdata.RowsNum;
        }
    }
}
