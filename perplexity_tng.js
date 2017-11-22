var stem = require('stem-porter');

var process = function(model, sentences, languages, alphaValue, randomSeed) {

  var W = model.hypers.W;
  var T = model.hypers.T;
  var vocab = model.hypers.vocab;
  var vocabOrig = model.hypers.vocabOrig;

  var _alpha = model.priors.alpha;
  var beta = model.priors.beta;
  var gamma = model.priors.gamma;
  var delta = model.priors.delta;

  var theta = model.posteriors.theta;
  var phi = model.posteriors.phi;
  var psi = model.posteriors.psi;
  var sigma = model.posteriors.sigma;
  
  var n_zw = model.counters.n_zw;
  var m_zwv = model.counters.m_zwv;
  var p_zwk = model.counters.p_zwk;
  var n_z = model.counters.n_z;
  var m_zw = model.counters.m_zw;
  
  // Result is perplexity of the model
  var result = 0;

  // Index-encoded array of sentences, with each row containing the indices of the words in the vocabulary.
  var documents = new Array();
  // Hash of vocabulary words and the count of how many times each word has been seen.
  var f = {};
  // Vocabulary of unique words in their original form.
  for(var i=0;i<vocab.length;i++) {
    f[vocab[i]] = 1;
  }
  // Array of stop words
  languages = languages || Array('en'); 
  if (sentences && sentences.length > 0) {
    var stopwords = new Array();

    languages.forEach(function(value) {
        var stopwordsLang = require('./stopwords_' + value + ".js");
        stopwords = stopwords.concat(stopwordsLang.stop_words);
    });

    for(var i=0;i<sentences.length;i++) {
      if (sentences[i]=="") continue;
      documents[i] = new Array();

      var words = sentences[i] ? sentences[i].split(/[\s,\"]+/) : null;
      console.log('words = ' +JSON.stringify(words));
      
      if(!words) continue;
      for(var wc=0;wc<words.length;wc++) {
        var w=words[wc].toLowerCase();
        if(languages.indexOf('en') != -1)
          w=w.replace(/[^a-z\'A-Z0-9\u00C0-\u00ff ]+/g, '');
        var wStemmed = stem(w);
        if (w=="" || !wStemmed || w.length==1 || stopwords.indexOf(w.replace("'", "")) > -1 || stopwords.indexOf(wStemmed) > -1 || w.indexOf("http")==0) continue;
        if (f[wStemmed]) { 
            f[wStemmed]=f[wStemmed]+1;
            documents[i].push(vocab.indexOf(wStemmed));
          } 
        else if(wStemmed) { 
          // We use -1 to indicate verbatim that is not existing in our model dictionary.
          documents[i].push(-1);
        };            
      }
    }

    var D = documents.length;
    var alpha = alphaValue || _alpha;  // per-document distributions over topics
    documents = documents.filter((doc) => { return doc.length }); // filter empty documents

    console.log('docs length = '+documents.length);


    console.log('Start calculating Perplexity...');
    tng_perplexity.configure(documents,vocab,W, 10, 2000, 100, 10, randomSeed);
    console.log('Start running left-to-right algorithm...');
    result = tng_perplexity.left_to_right(T, alpha, beta, gamma, delta,
      n_zw, m_zwv, p_zwk, n_z, m_zw
    );
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

var tng_perplexity = new function() {
    // model state variables
    var documents; 
    var T; // # of Topic
    var D; // # of Docs
    var W; // # of unique words

    // model hyper-priors
    var alpha, beta, gamma, delta;
    
    // .. temporary variable to speedup Gibbs sampling
    var n_zw,m_zwv,p_zwk,n_z,m_zw;        
    
    var THIN_INTERVAL = 20;
    var BURN_IN = 100;
    var ITERATIONS = 1000;
    var SAMPLE_LAG;
    var RANDOM_SEED;
    var dispcol = 0;
    var numstats=0;
    var vocab = [];
    this.configure = function (docs,vocab,w,iterations,burnIn,thinInterval,sampleLag,randomSeed
    ) {
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
        this.vocab = vocab;         
    }
    this.initialState = function (n_zw, m_zwv, p_zwk, n_z, m_zw) {
      this.n_zw = n_zw;
      this.m_zwv = m_zwv;
      this.p_zwk = p_zwk;
      this.n_z = n_z;
      this.m_zw = m_zw;
    }
    
    this.left_to_right = function (T, alpha, beta, gamma, delta, 
      n_zw, m_zwv, p_zwk, n_z, m_zw) {
        var i;
        this.T = T;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.delta = delta;
        
        this.initialState(n_zw, m_zwv, p_zwk, n_z, m_zw);
        
        var logNumParticles = Math.log(this.ITERATIONS);
        var totalLogLikelihood = 0;

        for(var m = 0;m < this.documents.length;m++) {
          console.log('Processing doc #'+m);
          var docLogLikelihood = 0;
          var particleProbabilities = new Array(this.ITERATIONS);
          for(var particle=0; particle<this.ITERATIONS; particle++) {
            console.log(' Processing particle #'+particle);
            particleProbabilities[particle] = 
              this.left_to_right_sampling(m);
          }          
          console.log('Finished all particle, averaing result of size = '+particleProbabilities[0].length);
          for(var position=0; position<particleProbabilities[0].length; position++) {
            var sum = 0;
            for(var particle=0; particle<this.ITERATIONS; particle++) {
              sum += particleProbabilities[particle][position];
            }
            console.log('prob sum = '+sum);
            if(sum > 0.0) {
              var logProb = Math.log(sum) - logNumParticles;
              docLogLikelihood += logProb;
              //console.log(':: w='+this.documents[m][position]+', prob='+logProb);
            }
          }          
          totalLogLikelihood += docLogLikelihood;
          console.log(' - Document Log Likelihood = '+docLogLikelihood);          
        }
        return totalLogLikelihood;
    };

    this.left_to_right_sampling = function(m) {
      var docLength = this.documents[m].length;
      var wordProbabilities = makeArray(docLength);

      // Create copy of stat counters for each iteration to a document
      // so it gets cleanup after evaluate each document.
      var z = makeArray(docLength);
      var x = makeArray(docLength);

      // .. temporary variable to speedup Gibbs sampling
      var n_zw = make2DArray(this.T,this.W); // # of time word w is assigned to topic z as unigram
      var m_zwv = make3DArray(this.T,this.W,this.W); // # of time word v is assigned to topic z as 2nd term of word w
      var p_zwk = make3DArray(this.T,this.W,2); // # of time bigram status k is assigned for previous word w of topic (of previous word) z
      var q_dz = make2DArray(this.D,this.T); // # of word with topic z in document d
      var n_z = makeArray(this.T); // # of times any token is assigned to topic z as unigram
      var m_zw = make2DArray(this.T,this.W); // # of time any token is assigned to topic z as 2nd term of word w
      var N_d = makeArray(this.D);

      for (var t = 0; t < this.T; t++) {
        n_z[t] = this.n_z[t];
        for (var w = 0; w < this.W; w++) {
          n_zw[t][w] = this.n_zw[t][w];
          p_zwk[t][w][0] = this.p_zwk[t][w][0];
          p_zwk[t][w][1] = this.p_zwk[t][w][1];
          m_zw[t][w] = this.m_zw[t][w];
          for (var v = 0; v < this.W; v++) {
            m_zwv[t][w][v] = this.m_zwv[t][w][v];
          }
        }
      }
      
      for(var limit=0;limit<docLength;limit++) {
        for(var position=0;position<limit;position++) {
          // Disregard words those are not in dictionary.
          //console.log('w = '+this.documents[m][position]);
          if(this.documents[m][position] == -1)
            continue;
          //console.log('re-sample on position: '+position);
          console.log('re-sample on position: '+position+', word='+this.vocab[this.documents[m][position]]);          
          this.sampleFullConditional(m, position, true,
            n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
          );      
        }
        if(this.documents[m][limit] == -1)
          continue;
        console.log('sample on position: '+limit+', word='+this.vocab[this.documents[m][limit]]);
        var prob = this.sampleFullConditional(m, limit, false,
          n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
        );
        wordProbabilities[limit] += prob;           
      }
      return wordProbabilities;
    };
    
    this.sampleFullConditional = function(m,n, is_resampling,
        n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
      ) {
        var wordProbabilities = 0;
        var word = null;
        var topic = null;
        var status = null;
        var prev_topic = null;
        var prev_word = null;
        var next_status = null;

        word = this.documents[m][n];
        if(n > 0) {
          prev_word = this.documents[m][n-1];
          prev_topic = z[n-1];        
        }
        if(n < N_d[d] - 1) {
          next_status = this.x_d_i[d][n+1];
        }
      
        if(is_resampling) {
          topic = z[n];
          status = x[n];
    
          if(status == 0) {
            n_zw[topic][word]--;
            n_z[topic]--;        
          } else {
            m_zwv[topic][prev_word][word]--;
            m_zw[topic][prev_word]--;
          } // if
          if(prev_topic != null) {
            p_zwk[prev_topic][prev_word][status]--;
            //console.log(' - resamp: p_zwk['+prev_topic+','+prev_word+','+status+']: '
            //  +(p_zwk[prev_topic][prev_word][status]+1)+' => '+(p_zwk[prev_topic][prev_word][status]) );
          } // if
          if(next_status != null) {
            p_zwk[topic][word][next_status]--;
          }
          q_dz[m][topic]--;
          //console.log('resamp - q_dz['+m+']['+topic+'] '+(q_dz[m][topic]+1)+' => '+q_dz[m][topic]);
        }

        // calculate each P(z,x)
        var P_zx = make2DArray(this.T, 2);
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {         
            // Skip case of bigram status for first token in document because it is invalid
            if(n == 0 && _x == 1)
              continue;
            // increase counter for topic, status of current word in which calculation based on.
            if(_x == 0) {
              n_zw[_z][word]++;
              n_z[_z]++;        
            } else {
              m_zwv[_z][prev_word][word]++;
              m_zw[_z][prev_word]++;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]++;
              //console.log('  - try: p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]-1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][word][next_status]++;
            }              
            q_dz[m][_z]++;
            //console.log('try - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]-1)+' => '+q_dz[m][_z]);
            
            /*
            console.log(
              ((x==0)?',n_zw[z][word]='+n_zw[z][word]:',m_zwv[z][prev_word][word]='+m_zwv[z][prev_word][word])+
              ((x==0)?',n_z[z]='+n_z[z]:',n_zw[z][prev_word]='+n_zw[z][prev_word])+
              ((prev_topic != null)?',p_zwk[prev_topic][prev_word][x]='+p_zwk[prev_topic][prev_word][x]:'')+
              ((next_status != null)?',p_zwk[z][word][next_status]='+p_zwk[z][word][next_status]:'')+
              ',q_dz[m][z]='+q_dz[m][z]
            );
            */
              
            var first_term = 0.0;
            if(prev_topic != null) {
              first_term = (this.gamma + p_zwk[prev_topic][prev_word][_x] - 1)
                * (this.alpha + q_dz[m][_z] - 1);
              //console.log('ft[0]: '+first_term+', this.p_zwk[prev_topic][prev_word][x] = '+this.p_zwk[prev_topic][prev_word][x]);
            } else {
              first_term = (this.gamma)
                * (this.alpha + q_dz[m][_z] - 1);
              //console.log('ft[1]: '+first_term);
            }
  
            var second_term = null;
            if(_x == 0) {
              second_term = 
                (this.beta + n_zw[_z][word] - 1)
                / (this.W * this.beta + n_z[_z] - 1);
            } else {
              second_term = 
                (this.delta + m_zwv[_z][prev_word][word] - 1)
                / (this.W * this.delta + m_zw[_z][prev_word] - 1);
            }
  
            //console.log('first_term = '+first_term+', second_term = '+second_term);
            P_zx[_z][_x] = first_term * second_term;
            wordProbabilities += P_zx[_z][_x];
            
            // decrease counter back.
            if(_x == 0) {
              n_zw[_z][word]--;
              n_z[_z]--;        
            } else {
              m_zwv[_z][prev_word][word]--;
              m_zw[_z][prev_word]--;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]--;
              //console.log('  - roll: p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]+1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][word][next_status]--;
            }
            q_dz[m][_z]--;            
            //console.log('roll - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]+1)+' => '+q_dz[m][_z]);            
          } // for x
        } // for z
            
        // Sampling new topic, status from calculated P_zx
        // Note that if (token == 0) then x is forced to be 0...
        // Sampling in such case has to ignore case of x=1 and sampling from the rest probability.
        //console.log('Sampling: Doc['+document+']['+token+']');
        var sum = 0;
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {          
            sum = sum + P_zx[_z][_x];
            P_zx[_z][_x] = sum;
            //console.log(' - acc P_zx['+z+']['+x+'] = '+P_zx[z][x]);
          } // for x
        } // for z
        var u = this.getRandom() * sum;
        var new_topic = null;
        var new_status = null;
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {
            console.log('[z][x]='+_z+',',x+': p='+P_zx[_z][_x]);          
            if(u < P_zx[_z][_x]) {
              new_topic = _z;
              new_status = _x;
              break;
            } // if
          } // for x
          if(new_topic != null)
            break;
        } // for z
        topic = new_topic;
        status = new_status;

        z[n] = topic;
        x[n] = new_status;
        
        // Update all data, counter based on our sampling result
        if(status == 0) {
          n_zw[topic][word]++;
          n_z[topic]++;        
        } else {
          console.log('topic='+topic+',prevw='+prev_word+',w='+word);
          m_zwv[topic][prev_word][word]++;
          m_zw[topic][prev_word]++;
        } // if
        if(prev_topic != null) {
          p_zwk[prev_topic][prev_word][status]++;
          //console.log('  - set: p_zwk['+prev_topic+','+prev_word+','+status+']: '
          //  +(p_zwk[prev_topic][prev_word][status]-1)+' => '+(p_zwk[prev_topic][prev_word][status]) );
        } // if
        if(next_status != null) {
          p_zwk[topic][word][next_status]++;
        }
        q_dz[m][topic]++;        
        //console.log('set - q_dz['+m+']['+topic+'] '+(q_dz[m][topic]-1)+' => '+q_dz[m][topic]);            
        
        N_d[m]++;
        return wordProbabilities;
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

module.exports = process;
