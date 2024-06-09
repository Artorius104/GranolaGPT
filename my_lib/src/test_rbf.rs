extern crate rand;

use rand::Rng;

pub struct MyRBF {
    d: Vec<i32>, // Vecteur pour stocker les dimensions des couches du réseau de neurones
    w: Vec<Vec<Vec<f64>>>, // Vecteur pour stocker les poids entre les couches
    l: usize, // Nombre total de couches dans le réseau
    x: Vec<Vec<f64>>, // Vecteur pour stocker les valeurs d'entrée
    deltas: Vec<Vec<f64>>, // Vecteur pour stocker les erreurs
    centers: Vec<Vec<f64>>, // Centres des neurones RBF
    sigma: Vec<f64>, // Écart-type des fonctions de base radiale
}

impl MyRBF {
    pub fn new(npl: &Vec<i32>) -> MyRBF {
        let mut rbf = MyRBF {
            d: npl.clone(),
            w: vec![],
            l: 2, // Couche cachée + output
            x: vec![],
            deltas: vec![],
            centers: vec![],
            sigma: vec![],
        };

        // Initialisation des poids entre les couches
        for l in 1..=rbf.l {
            rbf.w.push(Vec::new()); // Ajout d'une nouvelle couche de poids

            for _i in 0..=rbf.d[l] as usize {
                rbf.w[l - 1].push(Vec::new()); // Ajout d'une nouvelle ligne de poids

                for _j in 0..=rbf.d[l - 1] as usize {
                    if _j == 0 {
                        rbf.w[l - 1][_i].push(0.0); // Initialisation de poids à zéro pour le biais
                    } else {
                        let random_value: f64 = rand::thread_rng().gen(); // Génération d'un nombre aléatoire entre 0 et 1
                        rbf.w[l - 1][_i].push(random_value * 2.0 - 1.0); // Initialisation de poids aléatoires dans la plage [-1, 1]
                    }
                }
            }
        }

        // Initialisation des valeurs d'entrée et des erreurs
        for l in 0..=rbf.l {
            rbf.x.push(Vec::new()); // Ajout d'une nouvelle couche d'entrée
            rbf.deltas.push(Vec::new()); // Ajout d'une nouvelle couche d'erreurs

            for j in 0..=rbf.d[l] {
                rbf.deltas[l as usize].push(0.0); // Initialisation des erreurs à zéro

                if j == 0 {
                    rbf.x[l as usize].push(1.0); // Initialisation de la première valeur d'entrée à 1 pour le biais
                } else {
                    rbf.x[l as usize].push(0.0); // Initialisation des autres valeurs d'entrée à zéro
                }
            }
        }

        // Initialisation des centres et des sigmas pour les neurones RBF
        for _i in 0..npl[1] as usize {
            let mut center = Vec::new();
            for _ in 0..npl[0] as usize {
                let random_value: f64 = rand::thread_rng().gen_range(-1.0..=1.0);
                center.push(random_value);
            }
            rbf.centers.push(center);
            rbf.sigma.push(1.0); // Initialisation des sigmas à 1.0
        }

        rbf // Retour de la structure MyRBF initialisée
    }

    // Fonction de base radiale (Gaussienne)
    fn rbf(&self, x: &Vec<f64>, center: &Vec<f64>, sigma: f64) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += (x[i] - center[i]).powi(2);
        }
        (-sum / (2.0 * sigma.powi(2))).exp()
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        self.x[0] = input.clone();
        let hidden_dim = self.d[1] as usize;
        let output_dim = self.d[2] as usize;

        let mut hidden_outputs = vec![0.0; hidden_dim];
        for i in 0..hidden_dim {
            hidden_outputs[i] = self.rbf(&input, &self.centers[i], self.sigma[i]);
        }

        let mut output = vec![0.0; output_dim];
        for i in 0..output_dim {
            for j in 0..hidden_dim {
                output[i] += self.w[1][i][j] * hidden_outputs[j];
            }
        }
        output
    }

    pub fn train(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) {
        let output = self.forward(input.clone());

        let hidden_dim = self.d[1] as usize;
        let output_dim = self.d[2] as usize;

        for i in 0..output_dim {
            let error = target[i] - output[i];
            self.deltas[1][i] = error;

            for j in 0..hidden_dim {
                let rbf_output = self.rbf(&input, &self.centers[j], self.sigma[j]);
                self.w[1][i][j] += learning_rate * error * rbf_output;
            }
        }
    }
}

//Comparer avec les predictions du prof pour checker le modèle

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_network() {
        // Initialisation du réseau RBF
        let npl = vec![2, 4, 1]; // Exemple de dimensions (input, hidden, output)
        let mut rbf_network = MyRBF::new(&npl);

        // Entraînement avec des exemples fictifs
        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let learning_rate = 0.1;
        for _ in 0..1000 {
            for &(ref input, ref target) in &training_data {
                rbf_network.train(input.clone(), target.clone(), learning_rate);
            }
        }

        // Test du réseau après entraînement
        for &(ref input, ref target) in &training_data {
            let output = rbf_network.forward(input.clone());
            println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
        }

        // Assertions pour vérifier que le réseau a appris correctement
        for &(ref input, ref target) in &training_data {
            let output = rbf_network.forward(input.clone());
            assert!((output[0] - target[0]).abs() < 0.1);
        }
    }
}

// fn main() {
//     // Initialisation du réseau RBF
//     let npl = vec![2, 4, 1]; // Exemple de dimensions (input, hidden, output)
//     let mut rbf_network = MyRBF::new(&npl);
//
//     // Entraînement avec des exemples fictifs
//     let training_data = vec![
//         (vec![0.0, 0.0], vec![0.0]),
//         (vec![0.0, 1.0], vec![1.0]),
//         (vec![1.0, 0.0], vec![1.0]),
//         (vec![1.0, 1.0], vec![0.0]),
//     ];
//
//     let learning_rate = 0.1;
//     for _ in 0..1000 {
//         for &(ref input, ref target) in &training_data {
//             rbf_network.train(input.clone(), target.clone(), learning_rate);
//         }
//     }
//
//     // Test du réseau après entraînement
//     for &(ref input, ref target) in &training_data {
//         let output = rbf_network.forward(input.clone());
//         println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, output);
//     }
// }
