var stem = require('stem-porter');
var fs = require('fs');

//
// Based on javascript LDA implementation https://github.com/awaisathar/lda.js
// Original code based on http://www.arbylon.net/projects/LdaGibbsSampler.java
// Modified to be TNG model describe at: https://people.cs.umass.edu/~mccallum/papers/tng-icdm07.pdf
//
var process__ = function(exec_type, sentences, numberOfTopics, numberOfTermsPerTopic, languages, alphaValue, betaValue, gammaValue, deltaValue, randomSeed, dict_map, rev_dict_map) {
    // The result will be map of every internal state of the model
    var result = {};
    // Index-encoded array of sentences, with each row containing the indices of the words in the vocabulary.
    var documents = new Array();
    // Hash of vocabulary words and the count of how many times each word has been seen.
    var f = {};
    // Vocabulary of unique words (porter stemmed).
    var vocab=new Array();
    // Vocabulary of unique words in their original form.
    var vocabOrig = {};
    // Array of stop words
    languages = languages || Array('en'); 
    if (sentences && sentences.length > 0) {

      if(exec_type === 'map_provided') {
        documents = sentences;
        for(var word in dict_map) {
            vocab.push(word);
            vocabOrig[word] = word;
        }
      } else {       
        var stopwords = new Array();

        languages.forEach(function(value) {
            var stopwordsLang = require('./stopwords_' + value + ".js");
            stopwords = stopwords.concat(stopwordsLang.stop_words);
        });


        for(var i=0;i<sentences.length;i++) {
            if (sentences[i]=="") continue;
            documents[i] = new Array();

            var words = sentences[i].split(/[\s,\"]+/);
            console.log('words = ' +JSON.stringify(words));

            if(!words) continue;
            for(var wc=0;wc<words.length;wc++) {
                var w = words[wc].toLowerCase();
                if(languages.indexOf('en') != -1)
                    w=w.replace(/[^a-z\'A-Z0-9\u00C0-\u00ff ]+/g, '');
                var wStemmed = stem(w);
                //console.log('wStemmed = ' +JSON.stringify(wStemmed));

                if (w=="" || !wStemmed || w.length==1 || stopwords.indexOf(w.replace("'", "")) > -1 || stopwords.indexOf(wStemmed) > -1 || w.indexOf("http")==0) continue;
                if (f[wStemmed]) { 
                    f[wStemmed]=f[wStemmed]+1;
                } 
                else if(wStemmed) { 
                    f[wStemmed]=1; 
                    vocab.push(wStemmed);
                    vocabOrig[wStemmed] = w;
                };
                
                documents[i].push(vocab.indexOf(wStemmed));
            }
        }
      }
          
      var W = vocab.length;
      var D = documents.length;
      var T = parseInt(numberOfTopics);
      var alpha = alphaValue || 0.1;  // per-document distributions over topics
      var beta = betaValue || .01;  // per-topic distributions over unigram words
      var gamma = gammaValue || 0.5;  // bigram status distributions over topics x words
      var delta = deltaValue || .01;  // bigram second words distribution over topics x first words

      documents = documents.filter((doc) => { return doc.length }); // filter empty documents
      
      tng.configure(documents, W, 10000, 200, 100, 10, /*randomSeed*/1);
      tng.gibbs(T, alpha, beta, gamma, delta);

      var theta = tng.getTheta();
      var phi = tng.getPhi();
      var psi = tng.getPsi();
      var sigma = tng.getSigma();

      result.topicModel = {};

      result.topicModel.hypers = {};
      result.topicModel.hypers.W = W;
      result.topicModel.hypers.T = T;
      result.topicModel.hypers.vocab = vocab;
      result.topicModel.hypers.vocabOrig = vocabOrig;
      
      result.topicModel.priors = {};
      result.topicModel.priors.alpha = alpha;
      result.topicModel.priors.beta = beta;
      result.topicModel.priors.gamma = gamma;
      result.topicModel.priors.delta = delta;

      result.topicModel.posteriors = {};
      result.topicModel.posteriors.theta = theta;
      result.topicModel.posteriors.phi = phi;
      result.topicModel.posteriors.psi = psi;
      result.topicModel.posteriors.sigma = sigma;
  
      result.topicModel.counters = {};
      result.topicModel.counters.n_zw = tng.n_zw;
      result.topicModel.counters.m_zwv = tng.m_zwv;
      result.topicModel.counters.p_zwk = tng.p_zwk;
      result.topicModel.counters.n_z = tng.n_z;
      result.topicModel.counters.m_zw = tng.m_zw;
  
      result.printReadableOutput = function() {

        // TODO: May output to string instead
        // var text = '';
        
        //bigram status
        console.log('=Bigram Status=');
        for (var k = 0; k < psi.length; k++) {
          var things = new Array();
          console.log('Topic ' + (k + 1));
          for (var w = 0; w < psi[k].length; w++) {
            things.push(""+psi[k][w][1]+"_"+psi[k][w][0] + "_" + vocab[w]);
          }
          things.sort().reverse();          
          for (var w = 0; w < psi[k].length; w++) {
            var tokens = things[w].split("_");
            console.log(' '+tokens[2]+' bigram: '+parseInt(tokens[0]*100)+'%, unigram: '+parseInt(tokens[1]*100)+'%');
          }          
        }
  
        console.log('=Bigram Word Distribution=');
        for (var k = 0; k < sigma.length; k++) {
          console.log('Topic ' + (k + 1));
          var things = new Array();
          for (var w = 0; w < sigma[k].length; w++) {
            for (var v = 0; v < sigma[k][w].length; v++) {
              things.push(""+sigma[k][w][v]+"_"+vocab[w] + "_" + vocab[v]);              
            }
          }          
          things.sort().reverse();          
          for (var v = 0; v < things.length; v++) {
            var tokens = things[v].split("_");
            var prob = parseInt(tokens[0]*100);
            if(prob < 2)
              continue;
            console.log(' '+tokens[1]+' -> '+tokens[2]+': '+prob+'%');
          }            
        }
        
        //topics
        console.log('=Topic Distribution=');
        var topTerms=numberOfTermsPerTopic;
        for (var k = 0; k < phi.length; k++) {
          var things = new Array();
          console.log('Topic ' + (k + 1));
          for (var w = 0; w < phi[k].length; w++) {
            //console.log(" "+phi[k][w].toPrecision(2)+"_"+vocab[w] + "_" + vocabOrig[vocab[w]]);
            things.push(""+phi[k][w].toPrecision(2)+"_"+vocab[w] + "_" + vocabOrig[vocab[w]]);
          }
          things.sort().reverse();
          if(topTerms>vocab.length) topTerms=vocab.length;

          for (var t = 0; t < topTerms; t++) {
            var topicTerm=things[t].split("_")[2];
            var prob=parseInt(things[t].split("_")[0]*100);
            if (prob<2) continue;              
            console.log(topicTerm + ' (' + prob + '%)');              
          }

        }              
      };

    }
    
    return result;
}

function makeArray(x) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=0;
    }
    return a;
}

function make2DArray(x,y) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=new Array();
        for (var j=0;j<y;j++)
            a[i][j]=0;
    }
    return a;
}

function make3DArray(x,y,z) {
  var a = new Array();    
  for (var i=0;i<x;i++)  {
      a[i]=new Array();
      for (var j=0;j<y;j++) {
          a[i][j]=new Array();
          for (var k=0;k<z;k++) {
            a[i][j][k]=0;
          }            
      }
  }
  return a;
}

var tng = new function() {
    // var documents,z,nw,nd,nwsum,ndsum,thetasum,phisum,W,T,alpha,beta,gamma,delta; 

    // model state variables
    var documents;
    var T; // # of Topic
    var D; // # of Docs
    var W; // # of unique words
    var N_d; // # of words in document d
    var z_d_i; // topic assigned for word[i] of doc[d]
    var x_d_i; // bigram status between word[i-1]->word[i] of doc[d]
    // var w_d_i; // word[i] of doc[d], reduntant with this.documents

    // model hyper-priors
    var alpha, beta, gamma, delta;

    // .. temporary variable to speedup Gibbs sampling
    var n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw;

    // For smoothing results using averaging value of posteriors
    var thetasum;
    var phisum;
    var psisum;
    var sigmasum;
    
    // other hyper parameters
    var THIN_INTERVAL = 20;
    var BURN_IN = 100;
    var ITERATIONS = 1000;
    var SAMPLE_LAG;
    var RANDOM_SEED;
    var dispcol = 0;
    var numstats=0;

    this.configure = function (docs,w,iterations,burnIn,thinInterval,sampleLag,randomSeed) {
        this.ITERATIONS = iterations;
        this.BURN_IN = burnIn;
        this.THIN_INTERVAL = thinInterval;
        this.SAMPLE_LAG = sampleLag;
        this.RANDOM_SEED = randomSeed;
        this.documents = docs;
        this.W = w;
        this.D = docs.length;
        this.dispcol=0;
        this.numstats=0; 
    }
    this.initializeState = function () {
        var i;

        this.N_d = makeArray(this.D);
        this.z_d_i = new Array();
        this.x_d_i = new Array();
        // this.w_d_i = new Array();
        
        // .. temporary variable to speedup Gibbs sampling
        this.n_zw = make2DArray(this.T,this.W); // # of time word w is assigned to topic z as unigram
        this.m_zwv = make3DArray(this.T,this.W,this.W); // # of time word v is assigned to topic z as 2nd term of word w
        this.p_zwk = make3DArray(this.T,this.W,2); // # of time bigram status k is assigned for previous word w of topic (of previous word) z
        this.q_dz = make2DArray(this.D,this.T); // # of word with topic z in document d
        this.n_z = makeArray(this.T); // # of times any token is assigned to topic z as unigram
        this.m_zw = make2DArray(this.T,this.W); // # of time any token is assigned to topic z as 2nd term of word w

        for (d=0;d<this.D;d++) {
          this.N_d[d] = this.documents[d].length;
          this.z_d_i[d] = new Array();
          this.x_d_i[d] = new Array();

          var prev_w = null;
          var prev_z = null;
          for (var i = 0; i < this.N_d[d]; i++) {
            var w = this.documents[d][i];
            var z = parseInt(""+(this.getRandom() * this.T));                 
            var x = parseInt(""+(this.getRandom() * 2));
            if(i == 0) x = 0; // First word can only be unigram.                 

            this.z_d_i[d][i] = z;
            this.x_d_i[d][i] = x;       
            //console.log('src_zwx = '+z+','+w+','+x+' '+prev_z+','+prev_w);

            if(x == 0) {
              this.n_zw[z][w]++;
              this.n_z[z]++;
            } else {              
              this.m_zwv[z][prev_w][w]++;              
              this.m_zw[z][prev_w]++;              
            } // if
            if(i > 0) {
              this.p_zwk[prev_z][prev_w][x]++;
            } // if
            this.q_dz[d][z]++;

            prev_w = w;
            prev_z = z;
          } // for i
        } // for d
        /*
        for(var z=0;z<this.T;z++)
          for(var w=0;w<this.W;w++)
            for(var k=0;k<2;k++)
              console.log('init: p_zwk['+z+','+w+','+k+'] = '+this.p_zwk[z][w][k]);
        */

      }
    
    this.gibbs = function (T, alpha, beta, gamma, delta) {
        var i;
        this.T = T;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.delta = delta;
        if (this.SAMPLE_LAG > 0) {
          this.thetasum = make2DArray(this.documents.length, this.T);
          this.phisum = make2DArray(this.T, this.W);            
          this.psisum = make3DArray(this.T, this.W, 2);            
          this.sigmasum = make3DArray(this.T, this.W, this.W);            
          this.numstats = 0;
        }
        this.initializeState();
        //document.write("Sampling " + this.ITERATIONS
         //   + " iterations with burn-in of " + this.BURN_IN + " (B/S="
         //   + this.THIN_INTERVAL + ").<br/>");
        for (i = 0; i < this.ITERATIONS; i++) {
            console.log('ITERATIONS: '+i);            
            for(var d=0;d<this.D;d++) {
              for(var n=0;n<this.N_d[d];n++) {
                var topic = this.z_d_i[d][n];
                var word = this.documents[d][n];
                var status = this.x_d_i[d][n];
                var prev_topic = null;
                var prev_word = null;
                var next_status = null;
                if(n > 0) {
                  prev_word = this.documents[d][n-1];
                  prev_topic = this.z_d_i[d][n-1];        
                }
                if(n < this.N_d[d] - 1) {
                  next_status = this.x_d_i[d][n+1];
                }
                //console.log('sam_zwx = '+topic+','+word+','+status+' '+prev_topic+','+prev_word);
                
                this.sampling_condition_and_update(d, n, word, topic, status, prev_word, prev_topic, next_status)
              }
            }
            if ((i < this.BURN_IN) && (i % this.THIN_INTERVAL == 0)) {
                //document.write("B");
                this.dispcol++;
            }
            if ((i > this.BURN_IN) && (i % this.THIN_INTERVAL == 0)) {
                //document.write("S");
                this.dispcol++;
            }
            if ((i > this.BURN_IN) && (this.SAMPLE_LAG > 0) && (i % this.SAMPLE_LAG == 0)) {
                this.updateParams();
                //document.write("|");                
                if (i % this.THIN_INTERVAL != 0)
                    this.dispcol++;
            }
            if (this.dispcol >= 100) {
                //document.write("*<br/>");                
                this.dispcol = 0;
            }
        }
    }
    
    // .. function for each Gibbs sampling loop
    // .. calculate P(z,x) of current word condition of every other words and Docs
    // .. P(z,x) must be calculated for every possible value of z,x so we can sampling on it
    this.sampling_condition_and_update = function(document, token, word, topic, status, prev_word, prev_topic, next_status) {
      // decrease counter from existing topic, status of current word
      if(status == 0) {
        this.n_zw[topic][word]--;
        this.n_z[topic]--;        
      } else {
        this.m_zwv[topic][prev_word][word]--;
        this.m_zw[topic][prev_word]--;
      } // if
      if(prev_topic != null) {
        this.p_zwk[prev_topic][prev_word][status]--;
        //console.log(' - 0: p_zwk['+prev_topic+','+prev_word+','+status+']: '
        //  +(this.p_zwk[prev_topic][prev_word][status]+1)+' => '+(this.p_zwk[prev_topic][prev_word][status]) );
      } // if
      if(next_status != null) {
        this.p_zwk[topic][word][next_status]--;
      }
      this.q_dz[document][topic]--;

      // calculate each P(z,x)
      var P_zx = make2DArray(this.T, 2);
      for(var z=0;z<this.T;z++) {
        for(var x=0;x<2;x++) {          
          // Skip case of bigram status for first token in document because it is invalid
          if(token == 0 && x == 1)
            continue;
          // increase counter for topic, status of current word in which calculation based on.
          if(x == 0) {
            this.n_zw[z][word]++;
            this.n_z[z]++;        
          } else {
            this.m_zwv[z][prev_word][word]++;
            this.m_zw[z][prev_word]++;
          } // if
          if(prev_topic != null) {
            this.p_zwk[prev_topic][prev_word][x]++;
            //console.log('  - try: p_zwk['+prev_topic+','+prev_word+','+x+']: '
            //  +(this.p_zwk[prev_topic][prev_word][x]-1)+' => '+(this.p_zwk[prev_topic][prev_word][x]) );
          } // if
          if(next_status != null) {
            this.p_zwk[z][word][next_status]++;
          }              
          this.q_dz[document][z]++;
    
          var first_term = 0.0;
          if(prev_topic != null) {
            first_term = (this.gamma + this.p_zwk[prev_topic][prev_word][x] - 1)
              * (this.alpha + this.q_dz[document][z] - 1);
            //console.log('ft[0]: '+first_term+', this.p_zwk[prev_topic][prev_word][x] = '+this.p_zwk[prev_topic][prev_word][x]);
          } else {
            first_term = (this.gamma)
              * (this.alpha + this.q_dz[document][z] - 1);
            //console.log('ft[1]: '+first_term);
          }

          var second_term = null;
          if(x == 0) {
            second_term = 
              (this.beta + this.n_zw[z][word] - 1)
              / (this.W * this.beta + this.n_z[z] - 1);
          } else {
            second_term = 
              (this.delta + this.m_zwv[z][prev_word][word] - 1)
              / (this.W * this.delta + this.m_zw[z][prev_word] - 1);
          }

          //console.log('first_term = '+first_term+', second_term = '+second_term);
          P_zx[z][x] = first_term * second_term;

          // decrease counter back.
          if(x == 0) {
            this.n_zw[z][word]--;
            this.n_z[z]--;        
          } else {
            this.m_zwv[z][prev_word][word]--;
            this.m_zw[z][prev_word]--;
          } // if
          if(prev_topic != null) {
            this.p_zwk[prev_topic][prev_word][x]--;
            //console.log('  - roll: p_zwk['+prev_topic+','+prev_word+','+x+']: '
            //  +(this.p_zwk[prev_topic][prev_word][x]+1)+' => '+(this.p_zwk[prev_topic][prev_word][x]) );
          } // if
          if(next_status != null) {
            this.p_zwk[z][word][next_status]--;
          }
          this.q_dz[document][z]--;
        } // for x
      } // for z

      // Sampling new topic, status from calculated P_zx
      // Note that if (token == 0) then x is forced to be 0...
      // Sampling in such case has to ignore case of x=1 and sampling from the rest probability.
      //console.log('Sampling: Doc['+document+']['+token+']');
      var sum = 0;
      for(var z=0;z<this.T;z++) {
        for(var x=0;x<2;x++) {          
          sum = sum + P_zx[z][x];
          P_zx[z][x] = sum;
          //console.log(' - acc P_zx['+z+']['+x+'] = '+P_zx[z][x]);
        } // for x
      } // for z
      var u = this.getRandom() * sum;
      var new_topic = null;
      var new_status = null;
      for(var z=0;z<this.T;z++) {
        for(var x=0;x<2;x++) {          
          if(u < P_zx[z][x]) {
            new_topic = z;
            new_status = x;
            break;
          } // if
        } // for x
        if(new_topic != null)
          break;
      } // for z
      topic = new_topic;
      status = new_status;

      //console.log(' - Sampling => topic='+topic+', status='+status);
      
 
      // Update all data, counter based on our sampling result
      if(status == 0) {
        this.n_zw[topic][word]++;
        this.n_z[topic]++;        
      } else {
        this.m_zwv[topic][prev_word][word]++;
        this.m_zw[topic][prev_word]++;
      } // if
      if(prev_topic != null) {
        this.p_zwk[prev_topic][prev_word][status]++;
        //console.log('  - set: p_zwk['+prev_topic+','+prev_word+','+status+']: '
        //  +(this.p_zwk[prev_topic][prev_word][status]-1)+' => '+(this.p_zwk[prev_topic][prev_word][status]) );
      } // if
      if(next_status != null) {
        this.p_zwk[topic][word][next_status]++;
      }
      this.q_dz[document][topic]++;
      
      this.z_d_i[document][token] = topic;
      this.x_d_i[document][token] = status;
    }
    
    this.updateParams =function () {
      for (var m = 0; m < this.documents.length; m++) {
        var q_d = 0;
        for(var z=0;z<T;z++)
          q_d += this.q_dz[m][z]; // not executed frequently enough to consider caching          
        for (var k = 0; k < this.T; k++) {
          this.thetasum[m][k] += (this.q_dz[m][k] + this.alpha) / (q_d + this.T * this.alpha);
        }
      }
      for (var k = 0; k < this.T; k++) {
        for (var w = 0; w < this.W; w++) {
          this.phisum[k][w] += (this.n_zw[k][w] + this.beta) / (this.n_z[k] + this.W * this.beta);
        }
      }
      for (var z = 0; z < this.T; z++) {
        for (var w = 0; w < this.W; w++) {
          this.psisum[z][w][0] += 
          (this.gamma + this.p_zwk[z][w][0]) 
          / (2 * this.gamma + this.p_zwk[z][w][0] + this.p_zwk[z][w][1]);
  
          this.psisum[z][w][1] += 
            (this.gamma + this.p_zwk[z][w][1]) 
            / (2 * this.gamma + this.p_zwk[z][w][0] + this.p_zwk[z][w][1]);
        }
      }
      for (var z = 0; z < this.T; z++) {
        for (var w = 0; w < this.W; w++) {
          var m_zw = 0;
          for(var v=0;v<this.W;v++)
            m_zw += this.m_zwv[z][w][v];
          for(var v=0;v<this.W;v++) {
            this.sigmasum[z][w][v] += 
              (this.delta + this.m_zwv[z][w][v]) 
              / (this.W * this.delta + m_zw);
          }
        }
      }
      this.numstats++;
    }
    
    this.getTheta = function() {
      var theta = make2DArray(this.D, this.T); // multinomial distribution of topics for doc[d]      
      if (this.SAMPLE_LAG > 0) {
        for (var m = 0; m < this.documents.length; m++) {
          for (var k = 0; k < this.T; k++) {
            theta[m][k] = this.thetasum[m][k] / this.numstats;
          }
        }
      } else {
        for (var m = 0; m < this.documents.length; m++) {
          var q_d = 0;
          for(var z=0;z<T;z++)
            q_d += this.q_dz[m][z]; // not executed frequently enough to consider caching          
          for (var k = 0; k < this.T; k++) {
            theta[m][k] = (this.q_dz[m][k] + this.alpha) / (q_d + this.T * this.alpha);
          }
        }
      }
      return theta;
    }
    
    this.getPhi = function () {
      var phi = make2DArray(this.T, this.W); // multinomial distribution of words for topic[z]
      if (this.SAMPLE_LAG > 0) {
        for (var k = 0; k < this.T; k++) {
          for (var w = 0; w < this.W; w++) {
            phi[k][w] = this.phisum[k][w] / this.numstats;
          }
        }
      } else {
        for (var k = 0; k < this.T; k++) {
          for (var w = 0; w < this.W; w++) {
            phi[k][w] = (this.n_zw[k][w] + this.beta) / (this.n_z[k] + this.W * this.beta);
          }
        }
      }
      return phi;
    }

    this.getPsi = function() {
      var psi = make3DArray(this.T, this.W, 2); // multinomial distribution of topics for doc[d]      
      if (this.SAMPLE_LAG > 0) {
        for (var z = 0; z < this.T; z++) {
          for (var w = 0; w < this.W; w++) {
            for (var x = 0; x < 2; x++) {
              psi[z][w][x] = this.psisum[z][w][x] / this.numstats;
            }
          }
        }
      } else {
        for (var z = 0; z < this.T; z++) {
          for (var w = 0; w < this.W; w++) {
            psi[z][w][0] = 
            (this.gamma + this.p_zwk[z][w][0]) 
            / (2 * this.gamma + this.p_zwk[z][w][0] + this.p_zwk[z][w][1]);
    
            psi[z][w][1] = 
              (this.gamma + this.p_zwk[z][w][1]) 
              / (2 * this.gamma + this.p_zwk[z][w][0] + this.p_zwk[z][w][1]);
          }
        }
      }
      return psi;
    }

    this.getSigma = function() {
      //console.log(JSON.stringify(this.m_zwv));
      var sigma = make3DArray(this.T, this.W, this.W); // multinomial distribution of topics for doc[d]      
      if (this.SAMPLE_LAG > 0) {
        for (var z = 0; z < this.T; z++) {
          for (var w = 0; w < this.W; w++) {
            for (var v = 0; v < this.W; v++) {
              sigma[z][w][v] = this.sigmasum[z][w][v] / this.numstats;
            }
          }
        }
      } else {
        for (var z = 0; z < this.T; z++) {
          for (var w = 0; w < this.W; w++) {
            var m_zw = 0;
            for(var v=0;v<this.W;v++)
              m_zw += this.m_zwv[z][w][v];
            for(var v=0;v<this.W;v++) {
              sigma[z][w][v] = 
                (this.delta + this.m_zwv[z][w][v]) 
                / (this.W * this.delta + m_zw);
            }
          }
        }
      }
      return sigma;
    }
    
    this.getRandom = function() {
        if (this.RANDOM_SEED) {
            // generate a pseudo-random number using a seed to ensure reproducable results.
            var x = Math.sin(this.RANDOM_SEED++) * 1000000;
            return x - Math.floor(x);
        } else {
            // use standard random algorithm.
            return Math.random();
        }
    }
}

module.exports = process__;

if(process.argv.length >= 3 && process.argv[2]==='unittest') {
    // Unit testing function
    function unitTest_TNG() {
      /*
        var sentences = [
            'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
            'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
            'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
            'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
            'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
            'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
            'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
            'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
            'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
            'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
            'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
            'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
            'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
            'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
            'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
        ];
      */

        var doc_map = require('./coded_training_data.json');
        var dict_map = require('./out_dict_map.json');
        var rev_dict_map = require('./out_dict_rev_map.json');
                
        var result = process__('map_provided', doc_map, 4, 10, ['th'], 0.1, 0.01, 0.5, 0.01, 100, dict_map, rev_dict_map);
        console.log('MODEL RAW OUTPUT: ');
        console.log(JSON.stringify(result.topicModel));
        console.log('');
        console.log('MODEL INFORMATION OUTPUT: ');
        result.printReadableOutput();

        fs.writeFileSync('model_output.json', JSON.stringify(result, null, 4));           
      } 
    unitTest_TNG();
}
