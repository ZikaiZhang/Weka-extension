package weka.test;

import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.*;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.io.File;


public class LogisticTestSepArff {
	
	static int numFolds = 10;
	
	public static void main(String[] argv) throws Exception
	{
		//Load data set
		ArffLoader loader1 = new ArffLoader();
		loader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-adult-train.arff"));
		Instances dataTrain = loader1.getDataSet();
		
		ArffLoader loader2 = new ArffLoader();
		loader2.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-adult-test.arff"));
		Instances dataTest = loader2.getDataSet();
		
		//prepare to evaluate
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);
		Evaluation eval = new Evaluation(dataTrain);

		//prepare the classifier
		Logistic LogisticTest = new Logistic();
		LogisticTest.buildClassifier(dataTrain);
		
		//evaluate
		eval.evaluateModel(LogisticTest, dataTest);
		
		//output
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toClassDetailsString("\nDetailed Statistics\n"));
		System.out.println(eval.toMatrixString("\nConfusion Matrix\n"));
	}
}
