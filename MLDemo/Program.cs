using System;
using System.Diagnostics;
using SimpleML;
using SimpleML.DataSet;

namespace MLDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var sw = new Stopwatch();
            sw.Start();

            var mnistdata = new MNIST(@".\Data\train-images.idx3-ubyte",
                @".\Data\train-labels.idx1-ubyte", 60000);
            var tupledata = mnistdata.GetDataSet();

            var mnisttest = new MNIST(@".\Data\t10k-images.idx3-ubyte",
                @".\Data\t10k-labels.idx1-ubyte", 10000);
            var tupletest = mnisttest.GetDataSet();

            Console.WriteLine("Data Loaded");
            Console.WriteLine($"Time Elapsed: {sw.Elapsed}\n");
            sw.Restart();

            var outputlabel = new[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
            var network = new NeuralNetwork(784, new[] { 20, 20 }, outputlabel, 1, 30, 0.2);
            
            for (var i = 0; i < 30; i++)
            {
                network.Train(tupledata.Item1, tupledata.Item2, false);

                var score = network.Score(tupletest.Item1, tupletest.Item2);
                Console.WriteLine($"Score(round {i}): {score * 100}% ({score * 10000}/10000)");

                Console.WriteLine($"Time Elapsed: {sw.Elapsed}\n");
                sw.Restart();
            }

            Console.ReadLine();
        }
    }
}
