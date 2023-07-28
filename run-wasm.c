#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#define WASM_EXPORT(name) __attribute__((used)) __attribute__((export_name(#name)))
#define WASM_IMPORT(module, name) __attribute__((used)) __attribute__((import_module(#module))) __attribute__((import_name(#name)))

WASM_IMPORT(env, print)
void env_print(const char * str);

WASM_IMPORT(env, exit)
void env_exit(int code);

WASM_IMPORT(env, random)
void env_random(void* buffer, uint32_t size);

static void printf_impl(const char* fmt, ...) {
    va_list argptr;
    static char buffer[1024];
    va_start(argptr, fmt);
    vsprintf(buffer, fmt, argptr);
    va_end(argptr);
    env_print(buffer);
}

#undef RAND_MAX
#define RAND_MAX 0x7FFFFFFF

static int rand_buffer[256];
static int rand_index = 256;

static int rand_impl() {
    if (rand_index >= 256) {
        rand_index = 0;
        env_random(rand_buffer, sizeof(rand_buffer));
    }
    return rand_buffer[rand_index++] & RAND_MAX;
}

#define exit env_exit
#define printf printf_impl
#define rand rand_impl

//#define LOG(format, ...) fprintf(stderr, "\n\e[1;31m" format "\e[0m\n", ##__VA_ARGS__)
#define LOG(format, ...)


// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

#include "out-220k/model.bin.c"

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;



void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
    float* ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    int head_size = p->dim / p->n_heads;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    
    // a few convenience variables
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim*sizeof(*x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
    
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++) {
            // get the q and k vectors for this head
            float* q = s->q + h * head_size;
            float* k = s->k + h * head_size;
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i+=2) {
                float q0 = q[i];
                float q1 = q[i+1];
                float k0 = k[i];
                float k1 = k[i+1];
                float fcr = freq_cis_real_row[i/2];
                float fci = freq_cis_imag_row[i/2];
                q[i]   = q0 * fcr - q1 * fci;
                q[i+1] = q0 * fci + q1 * fcr;
                k[i]   = k0 * fcr - k1 * fci;
                k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));
        
        // multihead attention. iterate over all heads
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * dim + h * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);
            
            // weighted sum of the values, store back into xb
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    val += att[t] * s->value_cache[loff + t * dim + h * head_size + i]; // note bad locality
                }
                s->xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }
        
        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }
    
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------

char* create_word(const char* token, int postfix_ed, int postfix_ing, int postfix_s)
{
    static char buffer[1024];
    strcpy(buffer, token);
    char* end = buffer + strlen(buffer);
    if (postfix_s) {
        if (end[-1] == 's' || end[-1] == 'x' || end[-1] == 'z') {
            *end++ = 'e';
        }
        if (end[-1] == 'h' && (end[-2] == 's' || end[-2] == 'c')) {
            *end++ = 'e';
        }
        *end++ = 's';
    }
    if (postfix_ed) {
        if (end[-1] != 'e') {
            *end++ = 'e';
        }
        *end++ = 'd';
    }
    if (postfix_ing) {
        if (end[-1] == 'e') {
            end--;
        }
        *end++ = 'i';
        *end++ = 'n';
        *end++ = 'g';
    }
    *end++ = 0;
    return buffer;
}

static char prev[32] = "";
static char this[32];
static int capitalize = 0;
static int uppercase = 0;
static int postfix_ed = 0;
static int postfix_ing = 0;
static int postfix_s = 0;

char* decode_token(const char* token) {

    LOG("---------------------------------------------\ntoken: %s", token);

    static char buffer[1024];
    char* end = buffer;

    if (token[0] >= '0' && token[0] <= '9') {
        LOG("digit");
        strcpy(this, "0");
    } else if (isalpha(token[0])) {
        LOG("word");
        strcpy(this, "a");
    } else {
        LOG("other");
        strcpy(this, token);
    }
    
    if (strcmp("~1", token) == 0) {
        LOG("capitalize");
        capitalize = 1;
    } else if (strcmp("~2", token) == 0) {
        LOG("uppercase");
        uppercase = 1;
    } else if (strcmp("-s", token) == 0) {
        LOG("postfix_s");
        postfix_s = 1;
    } else if (strcmp("-ed", token) == 0) {
        LOG("postfix_ed");
        postfix_ed = 1;
    } else if (strcmp("-ing", token) == 0) {
        LOG("postfix_ing");
        postfix_ing = 1;
    } else {
        char x;
        if (prev[0] == 0 || strcmp(this, "'>") == 0 || strcmp(this, "\">") == 0) {
            //pass
            x = '1';
        } else if (strcmp(prev, "--") == 0) {
            *end++ = ' ';
            x = '2';
        } else if (strcmp(prev, "'<") == 0 || strcmp(prev, "\"<") == 0 || (this[1] == 0 && (this[0] == '-' || this[0] == '\'' || this[0] == '.' || this[0] == ',' || this[0] == '!' || this[0] == '?' || this[0] == ':'))) {
            //pass
            x = '3';
        } else if (strcmp(prev, "'>") == 0 || strcmp(prev, "\">") == 0 || (prev[1] == 0 && (prev[0] == 'a' || prev[0] == '.' || prev[0] == ',' || prev[0] == '!' || prev[0] == '?' || prev[0] == ':')) || strcmp(this, "--") == 0) {
            *end++ = ' ';
            x = '4';
        } else if ((prev[1] == 0 && (prev[0] == '\'' || prev[0] == '-')) || strcmp(this, "0") == 0) {
            //pass
            x = '5';
        } else {
            *end++ = ' ';
            x = '6';
        }
        strcpy(prev, this);

        if (strcmp(prev, "'")) {
            LOG("----------------%c", x);
        }

        if (strcmp(this, "a") == 0) {
            char* word = create_word(token, postfix_ed, postfix_ing, postfix_s);
            if (capitalize) {
                word[0] = toupper(word[0]);
            }
            if (uppercase) {
                int i;
                for (i = 0; i < strlen(word); i++) {
                    word[i] = toupper(word[i]);
                }
            }
            strcpy(end, word);
            end += strlen(word);
            LOG("Word: %s", word);
        } else if (strcmp("0", this) ||
                    strcmp(".", this) == 0 ||
                    strcmp(",", this) == 0 ||
                    strcmp("'", this) == 0 ||
                    strcmp("!", this) == 0 ||
                    strcmp("?", this) == 0 ||
                    strcmp("-", this) == 0 ||
                    strcmp(":", this) == 0 ||
                    strcmp("\"<", this) == 0 ||
                    strcmp("\">", this) == 0 ||
                    strcmp("--", this) == 0 ||
                    strcmp("'>", this) == 0 ||
                    strcmp("'<", this) == 0)
        {
            *end++ = token[0];
        } else {
            strcpy(end, token);
            end += strlen(token);
        }
        capitalize = 0;
        uppercase = 0;
        postfix_ed = 0;
        postfix_ing = 0;
        postfix_s = 0;
    }

    *end = 0;

    return buffer;
}

float* decompress_float() {
    static float buffer[sizeof(model_weights) / sizeof(model_weights[0])];
    uint32_t* dst = (uint32_t*)buffer;
    for (int i = 0; i < sizeof(model_weights) / sizeof(model_weights[0]); i++) {
        uint16_t value = model_weights[i];
        int negative = value & 0x8000;
        uint32_t exp = (value >> 10) & 0x1F;
        uint32_t frac = value & 0x3FF;
        if (exp > 0) {
            exp = exp - 24 + 127;
        }
        frac = frac << 13;
        *dst++ = frac | (exp << 23) | (negative ? 0x80000000 : 0);
    }
    return buffer;
}

static Config config = INIT_CONFIG;
static TransformerWeights weights;
static RunState state;
static int next;
static int token = 1;
static int pos = 0;

WASM_IMPORT(env, result)
void env_result(const char* text);

WASM_EXPORT(initialize)
void initialize() {

    //env_print("Initializing...");

    // read in the model.bin file
    float* data = NULL;
    {
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        float* weights_ptr = decompress_float();
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
    }

    // create and init the application RunState
    malloc_run_state(&state, &config);
}

WASM_EXPORT(generate)
void generate(int steps, float temperature) {

    //env_print("Generating...");

    steps += pos;

    // the current position we are in
    while (pos < steps || token != 1) {

        // forward the transformer to get logits for the next token
        transformer(token, pos % config.seq_len, &config, &state, &weights);

        // sample the next token
        if(temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
            // apply softmax to the logits to get the probabilities for next token
            softmax(state.logits, config.vocab_size);
            // we now want to sample from this distribution to get the next token
            next = sample(state.logits, config.vocab_size);
        }
        const char *part = decode_token(vocab[next]);
        if (part[0]) env_result(part);

        // advance forward
        token = next;
        pos++;
    }
}
