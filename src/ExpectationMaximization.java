import java.util.ArrayList;

public class ExpectationMaximization {

    private ArrayList<ArrayList<Double>> values;

    /**
     * constructor
     */
    public ExpectationMaximization(ArrayList<ArrayList<Double>> values) {
        this.values = values;
    }

    /**
     * run the expectation maximization algorithm
     * @param n the number of clusters to generate
     */
    public void run(int n) {
        //TO DO: if given number of clusters is zero, need to determine optimal number of clusters
        //use BIC mentioned in assignment description
        ArrayList<Gaussian> gaussians = getInitialGaussians(n);
        //stores the probabilities that each point belongs to each gaussian, where
        //each row represents a point
        //each column represents one of the gaussians/clusters
        double[][] probabilities = new double[values.size()][values.get(0).size()];
        long startTime = System.currentTimeMillis();
        //run while fewer than 10 seconds have elapsed
        double logLikelihood = 0;
        while (System.currentTimeMillis() - startTime < 10000) {
            probabilities = expectation(gaussians);
            //calculate log likelihood given these probabilities
            logLikelihood = logLikelihood(probabilities);
            //System.out.println(logLikelihood);
            //normalize probabilities and pass to maximization function
            probabilities = normalize(probabilities);
            gaussians = maximization(probabilities, gaussians.size());
        }
        //print out the cluster centers when done
        //TO DO: this method should return a list of clusters to be written to a file
        System.out.println("Cluster centers:");
        for (int i = 0; i < gaussians.size(); i++) {
            ArrayList<Double> mean = gaussians.get(i).getMean();
            System.out.println("Cluster " + (i + 1)  + ":");
            for (int j = 0; j < gaussians.get(0).getMean().size(); j++) {
                System.out.print(mean.get(j) + " ");
            }
            System.out.println();
        }
        System.out.println("Log likelihood: " + logLikelihood);
    }

    /**
     * generates n random gaussians
     * @param n number of gaussians to generate
     */
    private ArrayList<Gaussian> getInitialGaussians(int n) {
        ArrayList<Gaussian> gaussians = new ArrayList<Gaussian>();
        ArrayList<Integer> random = new ArrayList<Integer>();

        for (int i = 0; i < n; i++) { //generate n unique random numbers
            boolean valid = false;
            while (!valid) { //do this until we find a valid random number
                int r = (int) (Math.random() * values.size());
                if (!random.contains(r)) { //this random number is valid if we have not already used it
                    random.add(r);
                    valid = true;
                    //the 'rth' row of values is the cluster center for this gaussian
                    ArrayList<Double> mean = new ArrayList<Double>(values.get(r));

                    //set variance to a generic number
                    int size = mean.size();
                    ArrayList<Double> variance = new ArrayList<Double>();
                    for (int j = 0; j < size; j++) {
                        variance.add(5.0);
                    }

                    Gaussian g = new Gaussian(mean, variance);
                    gaussians.add(g);
                }
            }
        }

        return gaussians;
    }

    /**
     * given the probabilities of each point belonging to each cluster (not normalized),
     * calculate the log likelihood of the data
     */
    private double logLikelihood(double[][] probabilities) {
        double logLikelihood = 0;
        for (int i = 0; i < probabilities.length; i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[0].length; j++) {
                sum += probabilities[i][j];
            }
            sum = Math.log(sum);
            logLikelihood += sum;
        }
        return logLikelihood;
    }

    /**
     * runs the expectation step of the algorithm
     * given the cluster means and variances,
     * calculate the probability of each point belonging to each cluster
     */
    private double[][] expectation(ArrayList<Gaussian> gaussians) {
        double[][] probabilities = new double[values.size()][gaussians.size()];

        for (int i = 0; i < values.size(); i++) { //iterate through points
            ArrayList<Double> point = values.get(i);
            for (int j = 0; j < gaussians.size(); j++) { //repeat for each gaussian
                Gaussian gaussian = gaussians.get(j);
                double probability = 1;
                //need to multiply by value for each dimension of the point
                for (int k = 0; k < point.size(); k++) {
                    double value = point.get(k);
                    double mean = gaussian.getMean().get(k);
                    double variance = gaussian.getVariance().get(k);
                    probability *= expectationFunction(value, mean, variance);
                }
                probabilities[i][j] = probability;
            }
        }
        return probabilities;
    }

    private double[][] normalize(double[][] probabilities) {
        for (int i = 0; i < probabilities.length; i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[0].length; j++) {
                sum += probabilities[i][j];
            }
            for (int j = 0; j < probabilities[0].length; j++) {
                probabilities[i][j] /= sum;
            }
        }
        return probabilities;
    }

    private double expectationFunction(double value, double mean, double variance) {
        double n1 = 1/(Math.sqrt(2 * Math.PI * variance));
        double n2 = -(Math.pow((value-mean), 2))/(2*variance);
        double n3 = n1*Math.pow(Math.E, n2);
        return n3;
    }

    /**
     * runs the maximization step of the algorithm
     * given the probabilities of each point belonging to each cluster,
     * recalculate the cluster means and variances
     */
    private ArrayList<Gaussian> maximization(double[][] probabilities, int n) {
        ArrayList<Gaussian> gaussians = new ArrayList<Gaussian>();

        for (int i = 0; i < n; i++) { //do for each gaussian
            ArrayList<Double> means = new ArrayList<Double>();
            ArrayList<Double> variances = new ArrayList<Double>();
            for (int j = 0; j < values.get(0).size(); j++) { //for each dimension of the point
                double numerator1 = 0;
                double denominator = 0;
                //calculating new MEAN
                for (int k = 0; k < probabilities.length; k++) { //for each point
                    ArrayList<Double> point = values.get(k);
                    numerator1 += probabilities[k][i]*point.get(j);
                    denominator += probabilities[k][i];
                }
                double mean = numerator1/denominator;
                means.add(mean);
                double numerator2 = 0;
                //calculating new VARIANCE
                for (int k = 0; k < probabilities.length; k++) { //for each point
                    ArrayList<Double> point = values.get(k);
                    numerator2 += probabilities[k][i]*Math.pow(point.get(j)-mean,2);
                }
                double variance = numerator2/denominator;
                variances.add(variance);
            }
            Gaussian g = new Gaussian(means, variances);
            gaussians.add(g);
        }

        return gaussians;
    }

}