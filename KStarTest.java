package weka.test;

import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.*;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.io.File;


public class KStarTest {
	
	static int numFolds = 10;
	
	public static void main(String[] argv) throws Exception
	{
		//Load dataset
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/winequality-red.csv"));
		Instances data = loader.getDataSet();
		
		//prepare to evaluate
		data.setClassIndex(data.numAttributes() - 1);
		Evaluation eval = new Evaluation(data);

		//prepare the classifier
		KStar KStarTest = new KStar();
		
		//train classifier and evaluate
		eval.crossValidateModel(KStarTest, data, numFolds, new Random(1));
		
		//output
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString("\nConfusion Matrix\n"));
	}
}
