package weka.test;

import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.*;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.io.File;


public class REPTreeTestSep {
	
	static int numFolds = 10;
	
	public static void main(String[] argv) throws Exception
	{
		//Load data set
		CSVLoader loader1 = new CSVLoader();
		loader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-breast-cancer-train.csv"));
		Instances dataTrain = loader1.getDataSet();
		
		CSVLoader loader2 = new CSVLoader();
		loader2.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-breast-cancer-test.csv"));
		Instances dataTest = loader2.getDataSet();
		
		//prepare to evaluate
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);
		Evaluation eval = new Evaluation(dataTrain);

		//prepare the classifier
		REPTree REPTreeTest = new REPTree();
		REPTreeTest.buildClassifier(dataTrain);;
		
		//evaluate
		eval.evaluateModel(REPTreeTest, dataTest);
		
		//output
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toClassDetailsString("\nDetailed Statistics\n"));
		System.out.println(eval.toMatrixString("\nConfusion Matrix\n"));
	}
}