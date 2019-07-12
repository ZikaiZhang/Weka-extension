package weka.test;

import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.*;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.io.File;


public class WeightedMajorityAlgorithm1TestSep {
	
	static int numFolds = 10;
	
	public static void main(String[] argv) throws Exception
	{
		//Load data set
		CSVLoader loader1 = new CSVLoader();
		loader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/adult-train.csv"));
		Instances dataTrain = loader1.getDataSet();
		
		CSVLoader loader2 = new CSVLoader();
		loader2.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/adult-test.csv"));
		Instances dataTest = loader1.getDataSet();
		
		//prepare to evaluate
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);
		Evaluation eval = new Evaluation(dataTrain);

		//prepare the classifier
		WeightedMajorityAlgorithm1 WeightedMajorityAlgorithm1Test = new WeightedMajorityAlgorithm1();
		WeightedMajorityAlgorithm1Test.buildClassifier(dataTrain);;
		
		//evaluate
		eval.evaluateModel(WeightedMajorityAlgorithm1Test, dataTest);
		
		//output
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString("\nConfusion Matrix\n"));
	}
}