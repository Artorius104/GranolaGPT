use rand::Rng;

use crate::utils::chunk_vector;

#[no_mangle]
pub struct MyMLP {
    d: Vec<i32>, // Vecteur pour stocker les dimensions des couches du réseau de neurones
    w: Vec<Vec<Vec<f64>>>, // Vecteur pour stocker les poids entre les couches
    l: usize, // Nombre total de couches dans le réseau
    x: Vec<Vec<f64>>, // Vecteur pour stocker les valeurs d'entrée
    deltas: Vec<Vec<f64>>, // Vecteur pour stocker les erreurs
}

impl MyMLP {
    // Méthode pour créer une nouvelle instance de MyMLP
    pub fn new(npl: &Vec<i32>) -> MyMLP {
        // Initialisation des champs de la structure MyMLP
        let mut mlp = MyMLP {
            d: npl.clone(),
            w: vec![],
            l: npl.len() - 1, // Nombre total de couches
            x: vec![],
            deltas: vec![],
        };

        // Initialisation des poids entre les couches
        for l in 0..=mlp.l {
            mlp.w.push(Vec::new()); // Ajout d'une nouvelle couche de poids

            if l == 0 {
                continue;
            }

            for _i in 0..=mlp.d[l - 1] as usize {
                mlp.w[l].push(Vec::new()); // Ajout d'une nouvelle ligne de poids

                for _j in 0..=mlp.d[l] {
                    if _j == 0 {
                        mlp.w[l][_i].push(0.0); // Initialisation de poids à zéro pour le biais
                    } else {
                        let random_value: f64 = rand::thread_rng().gen(); // Génération d'un nombre aléatoire entre 0 et 1
                        mlp.w[l][_i].push(random_value * 2.0 - 1.0); // Initialisation de poids aléatoires dans la plage [-1, 1]
                    }
                }
            }
        }

        // Initialisation des valeurs d'entrée et des erreurs
        for l in 0..=mlp.l {
            mlp.x.push(Vec::new()); // Ajout d'une nouvelle couche d'entrée
            mlp.deltas.push(Vec::new()); // Ajout d'une nouvelle couche d'erreurs

            for j in 0..=mlp.d[l] {
                mlp.deltas[l].push(0.0); // Initialisation des erreurs à zéro

                if j == 0 {
                    mlp.x[l].push(1.0); // Initialisation de la première valeur d'entrée à 1 pour le biais
                } else {
                    mlp.x[l].push(0.0); // Initialisation des autres valeurs d'entrée à zéro
                }
            }
        }
        mlp // Retour de la structure MyMLP initialisée
    }

    // Foraward Propagation
    pub fn propagate(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) {
        // Mise à jour des valeurs d'entrée de la première couche
        for j in 0..sample_inputs.len() {
            self.x[0][j + 1] = sample_inputs[j];
        }

        // Calcul des valeurs d'activation pour chaque couche
        for l in 1..self.l + 1 {
            for j in 1..(self.d[l] as usize) + 1 {
                let mut total = 0.0;

                // Calcul de la somme pondérée des entrées pour chaque neurone
                for i in 0..(self.d[l - 1] as usize) + 1 {
                    total += self.w[l][i][j] * self.x[l - 1][i];
                }

                // Application de la fonction d'activation (tanh sauf pour la dernière couche si c'est une classification)
                if is_classification || l < self.l {
                    total = total.tanh();
                }

                // Mise à jour de la valeur d'activation du neurone
                self.x[l][j] = total;
            }
        }
    }

    // Méthode pour prédire les sorties pour un ensemble d'entrées donné
    pub fn predict(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification); // Propagation des entrées à travers le réseau
        self.x[self.l][1..].to_vec() // Retour des valeurs de sortie de la dernière couche
    }

    // Méthode pour entraîner le MLP sur un ensemble de données d'entraînement
    pub fn train(&mut self, 
        X_train: &Vec<Vec<f64>>, 
        y_train: &Vec<Vec<f64>>,
        alpha: f64, 
        nb_iter: usize, 
        is_classification: bool) -> Vec<f64> {

        let mut losses: Vec<f64> = Vec::new(); // Vecteur pour stocker les pertes pour chaque époque

        // Boucle d'entraînement
        for epochs in 0..nb_iter {
        let mut epoch_loss = 0.0; // Perte totale pour cette époque

        // Boucle sur chaque échantillon
        for k in 0..X_train.len() {
            let sample_inputs = &X_train[k];
            let sample_expected_outputs = &y_train[k];

            self.propagate(sample_inputs, is_classification); // Propagation des entrées à travers le réseau

            // Calcul de la perte (MSE) pour cet échantillon
            let mut sample_loss = 0.0;
            for j in 1..(self.d[self.l] as usize) + 1 {
                let error = self.x[self.l][j] - sample_expected_outputs[j - 1];
                sample_loss += error.powi(2);
                self.deltas[self.l][j] = error;

                if is_classification {
                    self.deltas[self.l][j] *= (1.0 - self.x[self.l][j].powi(2));
                }
            }
            epoch_loss += sample_loss / (self.d[self.l] as f64);

            // Backpropagation du gradient
            for l in (2..self.l).rev() {
                for i in 1..(self.d[l - 1] as usize) + 1 {
                    let mut total = 0.0;
                    for j in 1..(self.d[l] as usize) + 1 {
                        total += self.w[l][i][j] * self.x[l - 1][i];
                    }
                    total *= 1.0 - self.x[l - 1][i].powi(2);
                    self.deltas[l - 1][i] = total;
                }
            }

            // Mise à jour des poids
            for l in 1..self.l + 1 {
                for i in 0..(self.d[l - 1] as usize) + 1 {
                    for j in 1..(self.d[l] as usize) + 1 {
                        self.w[l][i][j] -= alpha * self.x[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
        }
        losses.push(epoch_loss / (X_train.len() as f64)); // Stockage de la perte pour cette époque normalisée par le nombre d'échantillons
        }
        losses // Retour des pertes pour chaque époque
    }
}

#[no_mangle]
pub extern "C" fn create_MyMLP(arr: *const i32, arr_size: i32) -> *mut MyMLP {
    let nlp = unsafe{ 
        std::slice::from_raw_parts(arr, arr_size as usize)
    };

    let model = MyMLP::new(&nlp.to_vec());

    let boxed_model = Box::new(model);
    let leaked_boxed_model = Box::leak(boxed_model);
    leaked_boxed_model
}

#[no_mangle]
pub extern "C" fn train_MyMLP(
                            p_model: *mut MyMLP,

                            p_X:*const f64,
                            X_chunk: i32,

                            p_y:*const f64,
                            y_chunk : i32,

                            n_samples: i32, 
                            alpha: f64, 
                            nb_iter: i32, 
                            is_classification: bool) -> *const f64{

    let model = unsafe {&mut *p_model};

    let X_flatten = unsafe {
        {std::slice::from_raw_parts(p_X, (X_chunk * n_samples) as usize)}
    };
    let X = chunk_vector(X_flatten.to_vec(), X_chunk as usize);

    let y_flatten = unsafe {
        {std::slice::from_raw_parts(p_y, (y_chunk * n_samples) as usize)}
    };
    let y = chunk_vector(y_flatten.to_vec(), y_chunk as usize);


    let losses = model.train(&X, &y, alpha, nb_iter as usize, is_classification);

    let leaked_losses = Vec::leak(losses.to_vec());
    leaked_losses.as_ptr()
}

#[no_mangle]
pub extern "C" fn predict_MyMLP(p_model:*mut MyMLP, 
                                is_classification: bool, 
                                p_samples:*const f64, 
                                samples_chunk:i32, 
                                n_samples:i32) -> *const f64{
    let mut predictions:Vec<f64> = vec![];
    let model = unsafe{&mut *p_model};

    let samples_flatten = unsafe {
        std::slice::from_raw_parts(p_samples, (samples_chunk * n_samples) as usize)
    };

    let samples = chunk_vector(samples_flatten.to_vec(), samples_chunk as usize);

    for sample in samples{
        let sample_prediction = model.predict(&sample, is_classification);
        for value in sample_prediction{
            predictions.push(value);
        }
    }

    let leaked_predictions = Vec::leak(predictions);
    leaked_predictions.as_ptr()
}

