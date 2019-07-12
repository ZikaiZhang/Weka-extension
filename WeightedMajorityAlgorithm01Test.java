package weka.test;

import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.*;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Random;
import java.io.File;


public class WeightedMajorityAlgorithm01Test {
	
	public static void main(String[] argv) throws Exception
	{
		// Load data set
		CSVLoader loader1 = new CSVLoader();
		loader1.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-breast-cancer-train.csv"));
		Instances dataTrain = loader1.getDataSet();
		
		CSVLoader loader2 = new CSVLoader();
		loader2.setSource(new File("/Users/zhangzikai/Desktop/weka-3-8-3/partial-breast-cancer-test.csv"));
		Instances structure = loader2.getStructure();
		
		// prepare to evaluate
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		structure.setClassIndex(structure.numAttributes() - 1);
		
		// prepare the classifier
		WeightedMajorityAlgorithm01 weightedMajorityAlgorithm01Test = new WeightedMajorityAlgorithm01();
		weightedMajorityAlgorithm01Test.buildClassifier(dataTrain);
		
		// update & test the ensemble model
		Instance current;
		while((current = loader2.getNextInstance(structure)) != null)
		{
			weightedMajorityAlgorithm01Test.updateClassifier(current);
			
			weightedMajorityAlgorithm01Test.updateStat(current);
		}
			
		// output the confusion matrix
		System.out.println(Arrays.deepToString(weightedMajorityAlgorithm01Test.getConfusionMatrix()).replace("], ", "]\n"));
	}
}
