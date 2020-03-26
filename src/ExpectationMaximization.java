import java.util.ArrayList;

public class ExpectationMaximization {

    private ArrayList<ArrayList<Double>> values;
    private static final double LIMIT = 0.1; //used to determine when to restart

    /**
     * constructor
     */
    public ExpectationMaximization(ArrayList<ArrayList<Double>> values) {
        this.values = values;
    }

    /**
     * starts the expectation maximization algorithm
     * does BIC first if n = 0
     * @param n the number of clusters to generate
     */
    public void start(int n) {
        long startTime = System.currentTimeMillis();
        if (n == 0) n = getBIC();
        long timeElapsed = System.currentTimeMillis() - startTime;
        Result result = runEM(10000 - timeElapsed, n);
        System.out.println("done");
    }

    /**
     * runs the expectation maximization algorithm
     * @param time how long to run the algorithm for (in ms)
     * @param n number of clusters
     */
    private Result runEM(long time, int n) {
        //randomly initialize gaussians
        ArrayList<Gaussian> gaussians = getInitialGaussians(n);
        //store the best results
        ArrayList<Gaussian> bestGaussians = gaussians;
        double bestLL = -1000000;
        //stores the probabilities that each point belongs to each gaussian, where
        //each row represents a point
        //each column represents one of the gaussians/clusters
        double[][] probabilities = new double[values.size()][values.get(0).size()];
        long startTime = System.currentTimeMillis();
        //run while fewer than input ms have elapsed
        double LL = 0;
        double previousLL = -1000000;
        while (System.currentTimeMillis() - startTime < time) {
            probabilities = expectation(gaussians); //EXPECTATION!
            //calculate log likelihood given these probabilities
            previousLL = LL;
            LL = getLogLikelihood(probabilities);
            //normalize probabilities and pass to maximization function
            probabilities = normalize(probabilities);
            gaussians = maximization(probabilities); //MAXIMIZATION!
            //if new LL is better than current best one, store it
            if (LL > bestLL) {
                bestGaussians = gaussians;
                bestLL = LL;
            }
            if (LL - previousLL < 0.1) gaussians = getInitialGaussians(n);
        }
        //return a Result object that stores gaussians and LL
        return new Result(bestGaussians, bestLL);
    }

    /**
     * runs bayesian information criterion (BIC)
     * to determine optimal number of clusters
     */
    private int getBIC() {
        ArrayList<Double> BICS = new ArrayList<Double>();
        for (int i = 1; i <= 20; i++) {
            Result result = runEM(250, i);
            System.out.println(result.getLL());
            double k = result.getGaussians().size() * (2 * values.get(0).size() + 1); //number of parameters
            double n = values.size(); //sample size
            double BIC = k * Math.log(n) - 2 * result.getLL();
            BICS.add(BIC);
        }
        for (int i = 0; i < 20; i++) {
            System.out.println(BICS.get(i));
        }
        return 3; //TO DO: return the optimal value
    }

    /**
     * finds the angle made by the segments p1p2 and p2p3
     */
    private double getAngle(double p1, double p2, double p3) {
        double p1p2 = Math.sqrt(1 + Math.pow(p2 - p1, 2));
        double p2p3 = Math.sqrt(1 + Math.pow(p3 - p2, 2));
        double p1p3 = Math.sqrt(4 + Math.pow(p3 - p1, 2));

        double cosa = (Math.pow(p1p3, 2) - Math.pow(p1p2, 2) - Math.pow(p2p3, 2))/(-2*p1p2*p2p3);
        double a = Math.acos(cosa);
        System.out.println(Math.toDegrees(a));

        return 0;
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
                    //the value at position r is a cluster center
                    ArrayList<Double> mean = new ArrayList<Double>(values.get(r));

                    //set variance to a generic number
                    //TO DO: Change to variance of entire data set? Starting variance shouldn't matter too much.
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
    private double getLogLikelihood(double[][] probabilities) {
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

    /**
     * calculate the probability that a value belongs to a gaussian with given mean and variance
     */
    private double expectationFunction(double value, double mean, double variance) {
        double n1 = 1/(Math.sqrt(2 * Math.PI * variance));
        double n2 = -(Math.pow((value-mean), 2))/(2*variance);
        double n3 = n1*Math.pow(Math.E, n2);
        return n3;
    }

    /**
     * given a list of groups of probabilities, normalize each row
     */
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

    /**
     * runs the maximization step of the algorithm
     * given the probabilities of each point belonging to each cluster,
     * recalculate the cluster means and variances
     */
    private ArrayList<Gaussian> maximization(double[][] probabilities) {
        ArrayList<Gaussian> gaussians = new ArrayList<Gaussian>();

        for (int i = 0; i < probabilities[0].length; i++) { //do for each gaussian
            ArrayList<Double> means = new ArrayList<Double>();
            ArrayList<Double> variances = new ArrayList<Double>();
            for (int j = 0; j < values.get(0).size(); j++) { //for each dimension of the point
                double numerator1 = 0;
                double denominator = 0;
                //CALCULATING NEW MEAN
                for (int k = 0; k < probabilities.length; k++) { //for each point
                    ArrayList<Double> point = values.get(k);
                    numerator1 += probabilities[k][i]*point.get(j);
                    denominator += probabilities[k][i];
                }
                double mean = numerator1/denominator;
                means.add(mean);
                double numerator2 = 0;
                //CALCULATING NEW VARIANCE
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