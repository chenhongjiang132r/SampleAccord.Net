using Accord.Controls;
using Accord.Statistics;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SampleAccord.Net
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] inputs =
            {
                new double[] {0, 0},
                new double[] {0.25, 0.25},
                new double[] {0.5, 0.5},
                new double[] {1, 1},
            };

            int[] outputs =
            {
                0,
                0,
                1,
                1,
            };

            //Train a Logistic Regression Model
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                MaxIterations = 100
            };
            var logit = learner.Learn(inputs, outputs);

            //Predict Output
            bool[] predictions = logit.Decide(inputs);

            //Plot Results
            ScatterplotBox.Show("Expected Results", inputs, outputs);
            ScatterplotBox.Show("Actual Logistic Regression Output", inputs, predictions.ToZeroOne());

            Console.ReadKey();
        }
    }
}
