using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using SimpleMath.Collections;
using SimpleMath;
using System.Linq;

namespace SimpleML.DataSet
{
    /* Source: http://yann.lecun.com/exdb/mnist/
        TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        The labels values are 0 to 9.

        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
     */
    public class MNIST : IDataSet
    {
        private readonly string _figuresPath;
        private readonly string _labelsPath;
        private readonly int _size;

        public MNIST(string figurespath, string labelspath, int size)
        {
            _figuresPath = figurespath;
            _labelsPath = labelspath;
            _size = size;
        }

        public (DoubleMatrix, string[]) GetDataSet()
        {
            using var fstream = new FileStream(_figuresPath, FileMode.Open);
            using var freader = new BinaryReader(fstream);

            using var lstream = new FileStream(_labelsPath, FileMode.Open);
            using var lreader = new BinaryReader(lstream);

            freader.ReadBytes(4 * 4);
            lreader.ReadBytes(4 * 2);

            var figurematrix = new DoubleMatrix(_size, 784);
            var labelarray = new string[_size];

            for (var i = 0; i < _size; i++)
            {
                var figure = freader.ReadBytes(784);
                var label = lreader.ReadByte();

                for (var j = 0; j < 784; j++)
                {
                    figurematrix[i, j] = figure[j] / 255.0;
                }

                labelarray[i] = label.ToString();
            }

            return (figurematrix, labelarray);
        }

        public static void BytesToPicture(byte[] bytes, string savepath)
        {
            var bitmap = new Bitmap(28, 28);

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    var rgbvalue = bytes[x + y * 28];
                    bitmap.SetPixel(x, y, Color.FromArgb(rgbvalue, rgbvalue, rgbvalue));
                }
            }

            bitmap.Save(savepath, ImageFormat.Png);
        }

        public static void BytesToPicture(DoubleMatrix matrix, string savepath)
        {
            var array = Matrix.Convert1DMatrixToArray(matrix);
            var bytes = array.Select(i => (byte)(i * 256)).ToArray();

            BytesToPicture(bytes, savepath);
        }
    }
}
