package weka.classifiers.meta;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SGD;
import weka.classifiers.trees.*;
import weka.core.*;
import weka.classifiers.lazy.KStar;

public class WeightedMajorityAlgorithm03 extends AbstractClassifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6966647484077286913L;

	private static final double EPSILON = 0.000001;

	public Classifier[] ensemble = new Classifier[3];
	
	// the length of symbolic experts (3 is for the count of selecting 2 out of 3, 1 is for the count of selecting 3 out of 3)
	private int ensembleComb = 3 + 1;

	private double beta = 0.5;
	
	public double[] weight = new double[ensemble.length + ensembleComb];
	
	private int numClass = 2;
	
	protected int[][] confusionMatrix = new int[numClass][numClass];
	
	private int numCorPrByOneOrMoEx = 0;
	
	// method for setting the experts
	public void setExperts()
	{
		ensemble[0] = new SGD();
		
		ensemble[1] = new J48();
		
		ensemble[2] = new Logistic();
	}
	
	// method for increasing the number of predications predicted correctly by at least one expert
	public void incNumCorPrByOneOrMoEx() {
		numCorPrByOneOrMoEx += 1;
	}
	
	// method for returning the number of predictions predicted correctly by at least one expert
	public int getNumCorPrByOneOrMoEx() {
		return numCorPrByOneOrMoEx;
	}
	
	// method for determining whether it is true that at least one expert predicts correctly (This mechanism for measuring overlapping could be updated)
	public boolean oneOrMoCor(Instance instance) throws Exception {
		
		boolean test = false;
		
		for (int i = 0; i < 3; i++) {
			test = test || almostEqual(ensemble[i].classifyInstance(instance), instance.classValue());
		}
		
		return test;
	}
	
	// method for building the weighted majority algorithm classifier
	public void buildClassifier(Instances instances) throws Exception
	{
		// build model for three experts on a specific data set
		this.setExperts();
		
		for (int i = 0; i < this.ensemble.length; i++)
		{
			this.ensemble[i].buildClassifier(instances);
		}
				
		// setting the initial weight of different experts of weighted majority algorithm
		for (int i = 0; i < ensemble.length + ensembleComb; i++)
		{
			weight[i] = 1.0;
		}
	}

	// method for the main algorithm of updating the weighted majority algorithm ensemble model
	public void updateClassifier(Instance instance, int i) throws Exception
	{	
		// array for storing predictions by Experts
		double[] predByExperts = new double[ensemble.length + ensembleComb];

		// get the array of predictions by experts
		predByExperts = this.getPredByExperts(instance).clone();
		
		// test
		this.writeFile("/Users/zhangzikai/Desktop/predByExperts.csv", predByExperts);
		// end of test
					
		// make a weighted sum of these predictions and make a prediction for the ensemble model based on it
		double weiSum = getWeiSum(predByExperts, weight);
			
		double pred = this.predByEnsemble(weiSum, weight);	
		
		// test
		this.writeFile("/Users/zhangzikai/Desktop/testOutput.csv", weight);
		// end of test
			
		// compare the prediction of the ensemble model with the fact
		if (almostEqual(pred, instance.classValue()) == false) /* I need to further investigate the indices returned by classValue() */
		{
			// if it were wrong, update the weights of the experts or symbolic experts who predicted wrong
			for (int index = 0; index < ensemble.length + ensembleComb; index++)
			{
				if (almostEqual(predByExperts[index], pred))
				{
					weight[index] = weight[index] * beta;
				}
			}
			
			double sumWeight = Arrays.stream(weight).sum();
				
			// test
			sumWeight = weight[0] + weight[1] + weight[2] + weight[3] + weight[4] + weight[5] + weight[6];
			// end of test
			
			for (int index = 0; index < ensemble.length + ensembleComb; index++)
			{
				weight[index] = 1.0 * 7 * weight[index] / sumWeight;
			}
		}
		
		// update the number of predictions made correctly by at least one expert
		if(oneOrMoCor(instance) == true) {
			incNumCorPrByOneOrMoEx();
		}
		
	}
	
	
	// method for getting prediction by ensemble model
	public double predByEnsemble(double weiSum, double[] weight) throws Exception
	{
		double sumWeight = Arrays.stream(weight).sum(); //not sure of the function of the method call here
				
		sumWeight = weight[0] + weight[1] + weight[2] + weight[3] + weight[4] + weight[5] + weight[6];
		
		double pred;
		
		if ((sumWeight / 2) > weiSum)
		{
			pred = 0.0;
		}
		else if ((sumWeight / 2) < weiSum)
		{
			pred = 1.0;
		}
		else
		{
			Random rand = new Random();
			
			pred = (double) (rand.nextInt(2));
		}
		
		return pred;
	}
	
	// method for getting prediction by experts
	public double[] getPredByExperts(Instance instance) throws Exception
	{
		double[] predByExperts = new double[ensemble.length + ensembleComb];
		
		double[][] distByExperts = new double[ensemble.length][numClass];
		/* there is a problem with this chunk of code
		for (int i = 0; i < ensemble.length; i++)
		{
			distByExperts[i] = ensemble[i].distributionForInstance(instance).clone(); 
		}	                                                                  
		                                                                      	
		for (int j = 0; j < ensemble.length; j++)
		{
			predByExperts[j] = distByExperts[j][0] > distByExperts[j][1] ? 0.0 : 1.0;
		}
		*/
		for (int j = 0; j < ensemble.length; j++)
		{
			predByExperts[j] = ensemble[j].classifyInstance(instance);
		}
		
		for (int k = ensemble.length; k < ensemble.length + ensembleComb; k++)
		{
			predByExperts[k] = this.predBySymExperts(k, predByExperts);
		}
		
		return predByExperts;
	}
	
	// method for getting predictions by symbolic experts
	public double predBySymExperts(int numComb, double [] predByExperts)
	{
		double pred = 0.0;
		
		switch (numComb)
		{
		case 3: pred = calPred2(1, 2, predByExperts); break;
		
		case 4: pred = calPred2(1, 3, predByExperts); break;
		
		case 5: pred = calPred2(2, 3, predByExperts); break;
		
		case 6: pred = calPred3(1, 2, 3, predByExperts); break;
		}
		
		return pred;
	}
	
	// method for getting combinatorics predictions by symbolic experts
	public double calPred2(int i, int j, double[] predByExperts)
	{
		double pred = 0.0;
		
		if (almostEqual(predByExperts[i - 1] + predByExperts[j - 1], 2.0)) 
		{
			pred = 1.0;
		}
		
		return pred;
	}
	
	private boolean almostEqual(double d1, double d2) 
	{
		return Math.abs(d1-d2) < EPSILON;
	}

	// method for getting combinatorics predictions by symbolic experts
	public double calPred3(int i, int j, int k, double[] predByExperts)
	{
		double pred = 0.0;
		
		if (almostEqual(predByExperts[i - 1] + predByExperts[j - 1] + predByExperts[k - 1], 3.0))
		{
			pred = 1.0;
		}
		
		return pred;
	}
	
	// method for getting the sum of an array
	public double getWeiSum(double[] newArray, double[] weight)
	{
		double sum = 0.0;
		
		for (int i = 0; i < newArray.length; i++)
		{
			sum += newArray[i] * weight[i];
		}
		
		return sum;
	}
	
	// method for classify an instance
	public double classifyInstance(Instance instance) throws Exception
	{
		// compute the predictions by experts
		double[] predByExperts = new double[ensemble.length];
		
		predByExperts = this.getPredByExperts(instance).clone(); // clone needed
		
		// compute the weighted sum of predictions by experts
		double weiSum = getWeiSum(predByExperts, weight);
		
		// compute the prediction by the ensemble model
		double pred = this.predByEnsemble(weiSum, weight);	
		
		return pred;
	}

    // method for getting the distribution of classes of an instance
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		//return the distribution of probabilities of classes
		double[] dist = new double[numClass];
		
		dist[(int) this.classifyInstance(instance)] = 1.0;
		
		return dist;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/* test method for writing double values to file failed
	public void writeFile(String filename, double[] doubleArray) throws IOException
	{
		DataOutputStream testOutput = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filename, true)));
		
		for(int i = 0; i < this.ensemble.length - 1; i++)
		{
			testOutput.writeDouble(doubleArray[i]); 
			
			testOutput.writeChar(',');
		}
		
		testOutput.writeDouble(doubleArray[this.ensemble.length - 1]);
		
		testOutput.close();
	}
	*/
	
	// method for updating statistics
	public void updateStat(Instance current) throws Exception
	{
		this.updateConfusionMatrix(current);
	}
	
	// method for updating the confusion matrix
	public void updateConfusionMatrix(Instance current) throws Exception
	{
		if(almostEqual(current.classValue(), 0.0))
		{
			if(almostEqual(current.classValue(), this.classifyInstance(current)))
			{
				this.confusionMatrix[0][0]++; // I do not know for sure which row represents which nominal value
			}
			else
			{
				this.confusionMatrix[0][1]++;
			}
		}
		else
		{
			if(almostEqual(current.classValue(), this.classifyInstance(current)))
			{
				this.confusionMatrix[1][1]++;
			}
			else
			{
				this.confusionMatrix[1][0]++;
			}
		}
	}
	
	// method for giving confusion matrix to caller
	public int[][] getConfusionMatrix()
	{
		return confusionMatrix;
	}
	
	// test method
	public void writeFile(String filename, double[] doubleArray) throws IOException
	{
		BufferedWriter testOutput = new BufferedWriter(new FileWriter(filename, true));
		
		testOutput.write(Utils.arrayToString(doubleArray) + '\n');
		
		testOutput.close();
	}
}


