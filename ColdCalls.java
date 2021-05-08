package smartinternz;

//////////////////////////////////////////// JAVA PACKAGE FOR ARRAY MODULATIONS ////////////////////////////////////////////
import java.util.Arrays;

//////////////////////////////////////////// WEKA PACKAGES THAT ARE REQUIRED TO PERFORM PREDICTION ////////////////////////////////////////////
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

//////////////////////////////////////////// PUBLIC CLASS ////////////////////////////////////////////
public class ColdCalls {
	
//////////////////////////////////////////// METHOD TO GET ALL DATA FROM THE SPECIFIED PARAMETER FILE ////////////////////////////////////////////
	public static Instances getInstances (String filename)
	{
		
		DataSource source;
		Instances dataset = null;
		try {
			source = new DataSource(filename);
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			
//////////////////////////////////////////// EXCEPTION CATCH TO AVOID ERRORS IN ERROR IN READING FILE ////////////////////////////////////////////
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		}
		
		return dataset;
	}

//////////////////////////////////////////// MAIN METHOD OF THE CLASS ////////////////////////////////////////////
	public static void main(String[] args) throws Exception{

//////////////////////////////////////////// GETTING TRAINING AND TESTING DATASET FROM LOCAL FOLDER ////////////////////////////////////////////
		Instances train_data = getInstances("E:\\A.Sai Preeth\\placements\\oracle\\numeric\\edit\\rem\\rem2\\rem3\\rem4\\carInsurance_train.arff");
		Instances test_data = getInstances("E:\\A.Sai Preeth\\placements\\oracle\\numeric\\edit\\rem\\rem2\\rem3\\rem4\\carInsurance_test.arff");

//////////////////////////////////////////// PRINTING THE NUMBER OF INSTANCES IN THE TRAINING DATASET ////////////////////////////////////////////
		System.out.println("Number of instances in training dataset");
		System.out.println(train_data.size());
		System.out.println("Number of instances in testing dataset");		
		System.out.println(test_data.size());
		
//////////////////////////////////////////// LOGISTIC REGRESSION OBJECT IS CREATED ////////////////////////////////////////////
		Classifier classifier = new weka.classifiers.functions.Logistic();

//////////////////////////////////////////// LOGISTIC REGRESSION IS APPLIED ON TRAINING DATASET AND TRAINED MODEL IS OBTAINED ////////////////////////////////////////////
		classifier.buildClassifier(train_data);
		
//////////////////////////////////////////// APPLYING THE TRAINED MODEL TO TESTING DATASET ////////////////////////////////////////////
		Evaluation eva = new Evaluation(train_data);
		eva.evaluateModel(classifier, test_data);
		
//////////////////////////////////////////// CREATING AND PRINTING CONFUSION MATRIX OF THE TESTING DATASET ////////////////////////////////////////////
		double confusion[][] = eva.confusionMatrix();
		System.out.println("Confusion matrix:");
		for (double[] row : confusion)
			System.out.println(	 Arrays.toString(row));
		System.out.println("-------------------");
		
//////////////////////////////////////////// PRINTING AREA UNDER ROC CURVE ////////////////////////////////////////////
		System.out.println("Area under the curve");
		System.out.println( eva.areaUnderROC(0));
		System.out.println("-------------------");
				
//////////////////////////////////////////// PRINTING RECALL VALUE ////////////////////////////////////////////
		System.out.print("Recall :");
		System.out.println(Math.round(eva.recall(1)*100.0)/100.0);
		
//////////////////////////////////////////// PRINTING PRECISION VALUE ////////////////////////////////////////////
		System.out.print("Precision:");
		System.out.println(Math.round(eva.precision(1)*100.0)/100.0);

//////////////////////////////////////////// PRINTING F1 SCORE VALUE ////////////////////////////////////////////
		System.out.print("F1 score:");
		System.out.println(Math.round(eva.fMeasure(1)*100.0)/100.0);
		
//////////////////////////////////////////// PRINTING ACCURACY VALUE////////////////////////////////////////////
		System.out.print("Accuracy:");
		double acc = eva.correct()/(eva.correct()+ eva.incorrect());
		System.out.println(Math.round(acc*100.0)/100.0);
		
//////////////////////////////////////////// CHECKING WHETHER PREDICTION IS DONE PROPERLY OR NOT ////////////////////////////////////////////
		System.out.println("-------------------");
		Instance predicationDataSet = test_data.get(2);
		double value = classifier.classifyInstance(predicationDataSet);
		System.out.println("Predicted label:");
		System.out.print(value);
		
		
	}

}