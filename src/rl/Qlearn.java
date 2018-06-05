package rl;

import java.util.Hashtable;
import java.util.Random;

public class Qlearn {
	public double epsilon = 0.1; // parametre epsilon pour \epsilon-greedy
	public double alpha = 0.2; // taux d'apprentissage
	public double gamma = 0.9; // parametre gamma des eq. de Bellman/

	// Suggestions
	public int actions[];
	public Hashtable<Tuple<Integer, Integer>, Double> q;

	// Constructeurs
	public Qlearn(int[] actions) {
		this.actions = actions;
		q = new Hashtable<Tuple<Integer, Integer>, Double>();
	}

	public Qlearn(int[] actions, double epsilon, double alpha, double gamma) {
		this.actions = actions;
		this.epsilon = epsilon;
		this.alpha = alpha;
		this.gamma = gamma;
		q = new Hashtable<Tuple<Integer, Integer>, Double>();
	}

	/**
	 * Renvoie le reward qui se trouve sur la case dans la direction de "action".
	 */
	private double getReward(int state, int action) {
		int id_state = state;
		int[] get_state = new int[12];
		for (int i = 11; i >= 0; i--) {
			get_state[i] = (int) Math.floor(id_state / Math.pow(5, i));
			id_state = (int) (id_state % Math.pow(5, i));
		}

		int nextCell;

		switch (action) {
		case 0: // N
			nextCell = (get_state[5]);
			break;
		case 1: // NE
			nextCell = (get_state[8]);
			break;
		case 2: // E
			nextCell = (get_state[9]);
			break;
		case 3: // SE
			nextCell = (get_state[10]);
			break;
		case 4: // S
			nextCell = (get_state[6]);
			break;
		case 5: // SW
			nextCell = (get_state[3]);
			break;
		case 6: // W
			nextCell = (get_state[2]);
			break;
		case 7: // NW
			nextCell = (get_state[1]);
			break;
		default:
			nextCell = -1;
			break;
		}

		switch (nextCell) {
		case 1:// wall
			return -5;
		case 2:// ghost
			return -100;
		case 4:// food
			return 50;
		default:// nothing
			return 0;
		}
	}

	public int chooseAction(int s) {
		int best = 0;
		if (Math.random() < this.epsilon) {
			Random random = new Random();
			int action = random.nextInt(8);
			Tuple<Integer, Integer> stateAction = new Tuple<>(s, action);

			if (!q.containsKey(stateAction)) {
				q.put(stateAction, 0.0);
			}

			return action;
		} else {
			double[] reward = new double[8];
			for (int action : this.actions) {
				Tuple<Integer, Integer> stateAction = new Tuple<Integer, Integer>(s, action);
				if (!q.containsKey(stateAction)) {
					q.put(stateAction, 0.0);
					reward[action] = getReward(s, action);
				} else {
					reward[action] = q.get(stateAction) + getReward(s, action);
				}

			}
			double bestReward = reward[best];
			for (int i = 0; i < 8; i++) {// argmax de qSet
				if (reward[i] > bestReward) {
					best = i;
					bestReward = reward[best];
				}
			}
		}
		return best;
	}
	
	/*
	 * Function to initialize Q if the state is unknown
	 */
	public void check_initiate(Tuple<Integer, Integer> key) {
		if (!q.containsKey(key)) {
			this.q.put(key, 0.);
		}
	}
	
	/**
	 * Q-learn update function
	 */
	public void update(int a, int state, int newState, double reward) {
		Tuple<Integer, Integer> last = new Tuple<Integer, Integer>(state, actions[a]);
		
		check_initiate(last);
		Double lastReward = q.get(last);

		Tuple<Integer, Integer> bestActionTuple =
				new Tuple<Integer, Integer>(newState, chooseAction(newState));

		check_initiate(bestActionTuple);
		Double bestActionReward = q.get(bestActionTuple);

		q.put(last, lastReward + alpha * (reward + gamma * bestActionReward - lastReward));
	}

	/**
	 * SARSA learn function
	 */
	public void learn(int a, int state, int newState, double reward, int newAction) {

		// Utilisation de check_initiate sur (s,a)
		Tuple<Integer, Integer> key = new Tuple<Integer, Integer>(state, a);
		check_initiate(key);

		// L'ancienne valeur de q
		double oldQ = this.q.get(key);

		// Utilisation de check_initiate sur (s',a')
		Tuple<Integer, Integer> newKey = new Tuple<Integer, Integer>(newState, newAction);
		check_initiate(newKey);

		// Mise à jour
		double result = oldQ + this.alpha * (reward + this.gamma * this.q.get(newKey) - oldQ);
		this.q.put(key, result);
	}
}
