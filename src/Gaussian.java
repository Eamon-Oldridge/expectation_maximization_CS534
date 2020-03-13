import java.util.ArrayList;

public class Gaussian {

    private ArrayList<Double> mean;
    private ArrayList<Double> variance;

    /**
     * constructor
     * create an n-dimensional gaussian
     */
    public Gaussian(ArrayList<Double> m, ArrayList<Double> v) {
        this.mean = m;
        this.variance = v;
    }

    public ArrayList<Double> getMean() { return mean; }
    public ArrayList<Double> getVariance() { return variance; }
    public void setMean(ArrayList<Double> m) { this.mean = m; }
    public void setVariance(ArrayList<Double> v) { this.variance = v; }

}
