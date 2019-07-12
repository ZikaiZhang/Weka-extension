package weka.test;

import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import java.util.Random;
import java.io.File;


public class REPTreeTest {
	
	static int numFolds = 10;
	
	public static void main(String[] argv) throws Exception
	{
		//Load dataset
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/adult.csv"));
		Instances data = loader.getDataSet();
		
		//prepare to evaluate
		data.setClassIndex(data.numAttributes() - 1);
		Evaluation eval = new Evaluation(data);

		//prepare the classifier
		REPTree REPTreeTest = new REPTree();
		
		//train classifier and evaluate
		eval.crossValidateModel(REPTreeTest, data, numFolds, new Random(1));
		
		//output
		System.out.println(eval.toSummaryString("\nResults\n\n", false));
		System.out.println(eval.toMatrixString("\nConfusion Matrix\n"));
	}
}