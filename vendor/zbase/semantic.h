#ifndef SCC_H__
#define SCC_H__


//      semantic.h
//      
//      Copyright 2011 Ishanu Chattopadhyay <ishanu.chattopadhyay@cornell.edu>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//       
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.
//      
//      
 
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <arpa/inet.h>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/utility.hpp>             // for boost::tie
#include <boost/graph/graph_traits.hpp>  // for boost::graph_traits
#include <boost/graph/strong_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/serialization/strong_typedef.hpp>
#include <boost/thread.hpp>  
#include <boost/thread/mutex.hpp>

#include <boost/algorithm/string.hpp>

#include <iterator>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <utility>
#include <sys/time.h>
#include <unistd.h>
#include "config.h"

using namespace std; 
using namespace boost;

#define _semantic_RESET      0    
#define _semantic_BOLD       1    
#define _semantic_UNDERSCORE 4    
#define _semantic_BLINK      5   
#define _semantic_REVERSE    7    
#define _semantic_CONC       8   
#define _semantic_BLACK     30    
#define _semantic_RED       31  
#define _semantic_GREEN     32   
#define _semantic_YELLOW    33   
#define _semantic_BLUE      34    
#define _semantic_MAGENTA   35    
#define _semantic_CYAN      36   
#define _semantic_WHITE     37   
#define _semantic_B_BLACK   40   
#define _semantic_B_RED     41    
#define _semantic_B_GREEN   42    
#define _semantic_B_YELLOW  43   
#define _semantic_B_BLUE    44  
#define _semantic_B_MAGENTA 45   
#define _semantic_B_CYAN    46    
#define _semantic_B_WHITE   47    

#define MESSAGE(x) SCC_UTIL__::_MESSAGE((x), __PRETTY_FUNCTION__)
/*!
  Boost typedefs from using boost::graph for graphviz output
*/
typedef boost::adjacency_list<vecS, vecS, directedS,
  property<vertex_name_t, std::string>,
  property<edge_name_t, std::string> > Digraph;

/*!
  Default length of symbolic streams generated
*/
const size_t LEN_DEFAULT=1000;
/*!
  A \b symbol is an \c unsigned \c int \n
  The complete set of symbols define the alphabet
  Note that strong type used implies that 
  use of type \c unsigned \c int is not allowed in place of \c symbol.
  Also, there are some initialization and assigment differences.
*/
BOOST_STRONG_TYPEDEF(unsigned int,symbol);
//typedef unsigned int symbol;

/*!
  A \b state is an \c int \n
  We will use \c -1 for no connection (case where a particular
  symbol is not defined from a given state. Hence \b state 
  cannot be an \c unsigned \c int
*/

typedef vector <symbol> symbol_list_;

typedef int state;

/*!
  \b \f$ \widetilde{\Pi}\f$ is the morph matrix of dimension
  \f$ \vert Q \vert \times \vert \Sigma \vert\f$, such that 
  \f$ \widetilde{\Pi}\vert_{ij}\f$ is the probability of generating 
  symbol \f$j\f$ from state \f$i\f$. It is implemented as a \c map between \c state and \c vector \c <double>. Each \c vector represents a row of the corresponding \b stochastic matrix, and hence must \b sum \b to \b unity.
*/
typedef std::map < state,  vector < double > > pitilde;

/*!
  \c map between \c symbol and \c state. It represents a row of the 
  connection matrix defining the graph of the probabilistic automata.
*/
typedef std::map < symbol, state > map_sym_state;
/*!
  \b connx is the data-type for representing the underlying graph of 
  a probabilistic automata. It is implemented as a \c map between \c state and \c map_sym_state. Each map element  represents a row of the corresponding connection  matrix, such that \c connx \c var[i][j] is the new \c state after symbol \f$j\f$ is fired from \c state \f$i\f$. 
*/
typedef std::map < state,  map_sym_state > connx;

/*!
  \b uiuidbl  is the data-type for representing matrices with \c double entries. 
  It is implemented as a \c map between \c unsigned \c int  
  and a \c map between \c unsigned \c int and a \c double.
*/
typedef std::map < unsigned int, map < unsigned int, double > > matrix_dbl; 


/*!
  Typedef for the n-ary tree of integer valued symbolic derivatives
*/
typedef  map <symbol_list_, map <symbol, unsigned int > > phi_data_type_;

/*!
  Typedef for the index positions of the history fragments
*/
typedef map <symbol_list_, map <symbol, vector < unsigned int >  > > pin_data_type_;

/*!
  Typedef for the n-ary tree of double valued symbolic derivatives
*/
typedef map <symbol_list_, vector <double> >  stoch_phi_data_type_;


//---------------------------------------------------------------
class Symbolic_string_ ; /* Forward declaration*/
//---------------------------------------------------------------




/*!
  \b PFSA is the class for representing  probabilistic finite state automata of PFSA. 
  It requires a  \f$ \widetilde{\Pi}\f$, a \c connx, a current state \c curr_state. 
  Additionally, a certain length of data is generated and housed as \c data. 
  Multiple overloaded constructors and a copy constructor is implemented. 
  The \c ostream operators << is overloaded from easy  printout of \c data. 
  Other overloaded operators are \b +, \b !, and \b *
*/
class PFSA{ 

 private:

  /*!
    \b Stochastic morph matrix with each row summing to unity
  */
  pitilde pit;
  /*!
    \b Stochastic  matrix with each row summing to unity, for XPFSAs
    Need to use a derived class at some point
    Note: alphabet size of \c Xpit may not match the \c alphabet
  */
  pitilde Xpit;
  /*!
    \b Connection  matrix with integer (\c state) entries
  */
  connx aut;
  /*!
    \b Current \c state
  */
  state curr_state;
  /*!
    Vector of \c symbols generated according to morph and 
    connection matrices, and of specified length
  */
  symbol_list_ data;

  /*!
    Symbol \c alphabet (number of columns of pit and aut)
  */
  symbol alphabet;

  /*!
    Number of \c states (number of rows of pit and aut)
  */
  unsigned int NUM_STATE;

  /*!
    probability vector representing stationary distribution on the states.
    The unique distribution for a strongly connected machine is returned.
    No guarantees if PFSA is not strongly connected
  */
  vector <double> Stationary_distribution;

  /*!
    Square transition probability matrix, where \f$ \Pi_{ij} =
    \sum_{k:\delta(i,k)=j}\widetilde{\Pi}_{ik} \f$. 
  */
  vector <vector <double> > PImat;

  /*!
    \f$ \Gamma \f$ matrices returned as a map. \f$\Gamma_\sigma \f$  for the alphabet symbol \f$ \sigma \f$ is a square matrix of size \f$ \vert Q \vert \times \vert Q \vert \f$, such that \f$\Gamma_\sigma \vert_{ij} \f$ is the probability of going from state i to state j via symbol \f$ \sigma \f$.
  */
  map < symbol, vector < vector < double > > > Gamma;

 public:

  PFSA(){};
  PFSA(pitilde, connx);                /*!-- Constructor 1  */
  PFSA(pitilde, connx, state);          /*!-- Constructor 2 */
  PFSA(pitilde, connx, state, size_t);   /*!-- Constructor 3 */
  PFSA(string);    /* !-- Constructor from file */

  PFSA(const PFSA& G);           /*!  copy constructor  */
  
  /**
     \b Assignmnet operator
  */
  PFSA& operator= (const PFSA &);

  /*!
    \b Accessor function for \c private \c data
  */
  symbol_list_ *getdata(){return &data;};

  /*! \b Scalar multiplication  operator to generate PFSA
    \f$G\alpha\f$ from given PFSA \f$G\f$ and double \f$\alpha\f$. */
  PFSA& operator* (double);

  /*! \b Inversion  \b operator ! to generate PFSA 
    \f$-G\f$ from given PFSA \f$G\f$. */
  PFSA& operator! ();

  /*!
    Direct product of probabilistic automata.
  */
  PFSA& operator| (PFSA &G);

  /*!
    semantic wen sum of PFSAs 
  */
  PFSA& operator+ (PFSA &G);

 
  /*!
    Synchronous composition of probabilistic automata.
    as defined in "Structural transformations of probabilistic finite state machines",
    I Chattopadhyay, A Ray. International Journal of Control 81 (5), 820-835, 2008
    Also, in addition, returns the largest strong component (using \b opearator ~)
  */
  PFSA& operator|| (PFSA &G);

  /*!
    Projective composition of probabilistic automata.
    as defined in "Structural transformations of probabilistic finite state machines",
    I Chattopadhyay, A Ray. International Journal of Control 81 (5), 820-835, 2008
    This opeartion modifies the RHS operand.
  */
  PFSA& operator >> (PFSA &G);

  /*!
    Projective composition of probabilistic automata. with ITER specified
    as defined in "Structural transformations of probabilistic finite state machines",
    I Chattopadhyay, A Ray. International Journal of Control 81 (5), 820-835, 2008
    This opeartion modifies the RHS operand.
  */
  PFSA&  sync_proj(PFSA &G, unsigned int ITER);

  /*!
    return strong component, empty if no strong componenets
  */
  PFSA& operator~ ();


  /*! \b Generate data from  PFSA \f$G\f$ with \b randomized 
    initial state and of provided length\n
    Return a \c Symbolic_string_ reference*/
  Symbolic_string_  gen_data(size_t);

  /*! \b Generate data from  PFSA \f$G\f$ with \b randomized 
    initial state and of default length LEN _DEFAULT\n
    Return a \c Symbolic_string_ reference*/
  Symbolic_string_  gen_data();

  /**
     Print out the PFSA pitilde and connx matrices
     Use option 'mat' to output in MATLAB compatible version
  */
  void mc_print( ostream & out = std::cout );

  /** 
      defining the << for printing generated stream
  */
  friend ostream& operator<< (ostream &, PFSA &);
  
  /*!
    \b Function for drawing graphviz graph of automaton via boost:graph library
    dot parameters can be set by using dot.cfg in the same directory
  */
  void drawGraph();
  /*!
    \b Function for drawing graphviz graph of automaton via boost:graph library
    dot parameters can be set by editing dot.cfg in the same directory
    Name of the .dot file and the image file produced is Inferred_mc.x Name of the
    config file used for reading dot parameters is dot.cfg (optional)
  */
  void drawGraph(string dotfile);

  /*!
    \b Function for drawing graphviz graph of automaton via boost:graph library
    dot parameters can be set by specifyng \c configfile in the same directory
    \c dotfile is the name of the .dot file produced. Name of the config file used
    for reading dot parameters is dot.cfg (optional)
  */
  void drawGraph(string dotfile, string configfile, string Title="");

  /*!
    \b Function for drawing graphviz graph of \b xGenESeSS automaton 
    via boost:graph library. dot parameters can be set by specifyng \c configfile
    in the same directory. \c dotfile is the name of the .dot file produced. 
    Name of the config file used for reading dot parameters 
    is dot.cfg (optional)
  */
  void drawGraphX(string dotfile, string configfile, string Title="");

  /*!
    Generate stationary distribution on PFSA states
  */
  void gen_Stationary();

  /*!
    Generate stationary distribution on PFSA states with specified number of iterations
  */

  void gen_Stationary(unsigned int ITERATIONS);

  /*!
    Generate transition probability matrix
  */
  void gen_PI();

  /*!
    Generate \f$ \Gamma \f$ matrices as a map
  */
  void gen_Gamma();

  /*!
    Accessor function for stationary distribution
  */
  vector <double>  &get_Stationary()
    {
      if (Stationary_distribution.empty())
	gen_Stationary();
      return Stationary_distribution;
    };


  /*!
    Accessor function for transition probability matrix
  */
  vector <vector <double> >  &get_PI()
    {
      if (PImat.empty())
	gen_PI();
      return PImat;
    };

  /*!
    Accessor function for Gamma matrices (as a map)
  */
  map < symbol, vector < vector < double > > >  &get_Gamma()
    {
      if (Gamma.empty())
	gen_Gamma();
      return Gamma;
    };



  PFSA& canonize();

  double entropy();

  /*!
    Set function for \c connx \b aut
  */
  void set_aut(connx &aut1)  ;

  /*!
    Set function for \c pitilde \b pit
  */
  void set_pit(pitilde &pit1)  ;

  /*!
    Set function for \c pitilde \b Xpit
  */
  void set_Xpit(pitilde &pit)  ;

  /*!
    Accessor function for \c connx \b aut
  */
  connx &get_aut()  ;

  /*!
    Accessor function for \c pitilde \b pit
  */
  pitilde &get_pit()  ;


  /*!
    Accessor function for \c pitilde \b Xpit
  */
  pitilde &get_Xpit()  ;

  /*!
    Run function to get state distribution by running
    symbol stream through a labelled-graph. No use of \c pit made.
    Simply normalized counts of state visits.
  */
  vector <double>  run_graph(symbol_list_ history);

  /*!
    Run function to get symbol distribution expected in future. 
    \b future-steps specify
    how many steps in future the distributions are requested.
  */
  vector <vector <double> >  run(symbol_list_ history, unsigned int future_steps);

  /*!
    Run function to get symbol distribution using Xpit expected in future. 
    \b future-steps specify
    how many steps in future the distributions are requested.
  */
  vector <vector <double> >  runX(symbol_list_ history, 
				  unsigned int future_steps);



  /*!
    Run function to get symbol distribution using Xpit expected in future. 
    \b future-steps specify
    how many steps in future the distributions are requested.
    Additional parameter \b initstate
  */
  vector <vector <double> >  runX(symbol_list_ history, 
				  unsigned int future_steps, 
				  state initstate);

  vector <vector <double> >  runX(symbol_list_ history, 
				  unsigned int future_steps, 
				  vector <double> initstate,
				  vector <double>& nextstte);

  /*!
    Run function to get symbol distribution using Xpit expected in future. 
    \b future-steps specify
    how many steps in future the distributions are requested.
    \b string  \ c disttypr specifies the initial distribution
    which may take values  \b uniform or \b stationary
  */

  vector <vector <double> >  runX(symbol_list_ history, 
				  unsigned int future_steps, 
				  string disttype);

  /*!
    Check if PFSA is empty 
  */
  bool empty(){return aut.empty();};


  /*!
    rename states to read columns first. used in + operator implementation
   */
  PFSA& transpose_name(unsigned int colnum);


  /*!
    compute log likelihood
   */
  double log_likelihood(const symbol_list_& s);

  /*!
    compute log likelihood for a set of sequences 
   */
  vector<double> log_likelihood(const vector<symbol_list_> & s);
};


//---------------------------------------------------------------
/*!
  \b Symbolic_string_ is the class for representing  symbolic strings. 
  It requires a   \c data, implemented as a \c vector of \c symbols, 
  and the alphabet size of the data string.  Multiple overloaded constructors 
  and a copy constructor is implemented. The \c ostream operators << is 
  overloaded from easy  printout of \c data. Other overloaded 
  operators are \b +, \b !, and \b ~
*/
class Symbolic_string_ 
{
  /** symbolic data stream*/
  symbol_list_ data;

  /** synchronizing string */
  symbol_list_ omega_syn;
  /** alphabet size in symbolic data stream*/
  symbol alphabet;

  /** partition function on continuous data domain\n
      The vector elements specify the partition boundaries*/
  vector < double > cont_dom_partition;

  /*!
    Data types to compute norm via computation of the 
    symbolic derivatives: Phi and Datapin
  */
  phi_data_type_ Phi;

  /*!
    Data types to compute norm via computation of the 
    symbolic derivatives: Phi and Datapin
  */
  pin_data_type_ Datapin;

 public:

  /*!
    \b  Distance \c double from the zero string or symbolic white noise.\n
    2 norm is used 
  */
  double norm;

  /*!
    \b Constructors: default, from \c vector of \c symbols, and 
    from \c Symbolic_string_ i.e. the copy constructor
  */
  Symbolic_string_(){};
  Symbolic_string_(symbol_list_&);
  Symbolic_string_(symbol_list_ &data_in, unsigned int);
  Symbolic_string_(Symbolic_string_ const &);

  /** Constructor for generating \c Symbolic_string_ from symbolic data
      in file named  \c string (first argument)
  */
  Symbolic_string_(string);

  /** Constructor for generating \c Symbolic_string_ from continuous data
      in \c vector \c double (first argument), using partition function 
      as specified by second  argument */
  Symbolic_string_(vector < double >,vector < double >);

  /** Constructor for generating \c Symbolic_string_ from continuous data
      in single column tabulated in text file \b filename \c string (first argument), using partition function 
      as specified by second  argument */
  Symbolic_string_(string,vector < double >);

  /** Constructor for generating \c Symbolic_string_ from 2D continuous data
      in two columns tabulated in text file \b filename \c string (first argument), using partition function 
      as specified by second  (X) and third (Y) arguments */
  Symbolic_string_(string, vector < double >, vector <double>);

  /*!
    \b Operator ~ adds symbolic white noise to data, 
    and returns a reference to the \b new \c Symbolic_string_
  */
  Symbolic_string_  operator~();  
  /*!
    \b Operator + sums two \c Symbolic_string_, and returns a 
    reference to the \b new \c Symbolic_string_
  */
  Symbolic_string_  operator+(const Symbolic_string_ &);  
  /*!
    \b Overloaded Operator + which takes a \c vector \c <symbol> as the second argument */
  Symbolic_string_  operator+(const symbol_list_ &);  
  /*!
    \b Operator ! inverts a  symbolic string, and returns a reference to 
    the \b new \c Symbolic_string_. Internally, the implementation needs 
    to add symbolic white noise to the input stream possibly multiple times,
    and hence makes use of the operator ~.
  */
  Symbolic_string_  operator!();  /*<!-- semantic inversion -->*/


  /*!
    \b Operator * transforms  \c Symbolic_string_ via a cross PFSA to a \b new \c Symbolic_string_
  */
  Symbolic_string_  operator*(PFSA &);  




  /*!
    \b Operator << overloaded for direct pretty printing of \c data
  */
  friend ostream& operator<< (ostream &, Symbolic_string_ );

  /*!
    \b Accessor function for private member \b alphabet
  */
  symbol get_alphabet(){return alphabet;};

  symbol_list_ get_symbol_list(){return data;};

  /** Function to extract
      distance from zero string in the Wen sense
  */
  void get_norm();

  /** Function to extract
      distance from zero string considering all substrings upto \c DEPTH
  */
  void get_norm(unsigned int DEPTH);
  void get_norm_new(unsigned int DEPTH);
  void get_reflected_norm(unsigned int DEPTH);

  /** Computation of symbolic derivatives */
  void gen_Phi(symbol_list_ );

  /** Fast implementation of phi calculation with no recursive call back.
      Assumes that phi(omega0) exists, and return false on empty
      whre omega=omega0 sigma */
  bool gen_Phi_FAST(symbol_list_ omega);


  /** Memory management: cleaning up Phi entries. 
      Remove all entries with key equal or less length to slen */
  void Phi_clean(unsigned int slen);


  /** Memory management: cleaning up Datapin entries 
      Remove all entries with key equal or less length to slen */
  void Datapin_clean(unsigned int slen);

  /*!
    acessor function for symbolic derivatives
  */
  phi_data_type_ &get_Phi(){return Phi;};

  /*!
    acessor function for symbolic derivatives
  */
  pin_data_type_ &get_Pin(){return Datapin;};

  /** utility function to increment symbols */
  symbol_list_ & inc(symbol_list_ &s);

  /*!
    Functions ro compute synchronizing string.
    \c double \b  MIN_DEV_SYNC_SEARCH_ specifies the minimum deviation 
    beyond which we assume no improvement.
    \c unsigned int \b  LOOP_COUNT_SYNC_SEARCH_ specifies the max depth to which 
    synchronizing strings are serached for, and \b SYN_PREF_LENGTH_ specifies 
    the max length of different prefixes after which synchroninzing 
    strings are sought
  */
  symbol_list_ &getSync(double MIN_DEV_SYNC_SEARCH_,
			unsigned int SYN_PREF_LENGTH_ );
  void getSync();

  /** Set omega_syn manually */
  void setSync(symbol_list_ omega){omega_syn=omega;};

  /**   */
  symbol_list_ getSyncstring(){return omega_syn;};
  /*! Compute entropy of underlying process */

  double entropy();

  /*! Compute entropy of underlying process 
    where \c unsigned int \b MAX_DEPTH is the number of lexicographically ordered 
    strings (after considering the synchronizing prefix) that are considered in the computation of the entropy. Larger is better, wityh higher computational cost.
  */

  double entropy(unsigned int MAX_DEPTH, 
		 double EPSILON, int MIN_OCC=10);
};

//---------------------------------------------------------------
//---------------------------------------------------------------

void  get_Symbolic_DataMatrix(string,char, vector < symbol_list_ >*);
void  get_Symbolic_DataMatrix(string,char,unsigned int, vector < symbol_list_ >*);


/*!
  To ensure compatibility with the Gillespie run dataformat 

  [ time xval yval delxval delyval delforxval delforyval ]

  we take in file sequence and column number in file to treat as data.
  This function only deals in scalar time series, so only one of the columns is 
  read in as specified by the column number.

  How do we readin file sequence:

  1. Assume file name is of the form: <num>.txt, where num has a fixed width.
  2. So we need to specify begnum, endnum, fixedwidth
  3. Specify the #columnnumber column to read (where numbering starts with 0)
  4. Specify the partition to convert to symbols
  5. Specify maximum symbol stream length SZ
  6. Reference to container to store symbol stream matrix
  Symbolic_string_::Symbolic_string_(unsigned int begnum, 
  unsigned int endnum, 
  unsigned int fixedwidth,
  unsigned int columnmumber,
  unsigned int SZ,
  data matrix v_sym
*/

void  get_continuous_DataMatrix(unsigned int begnum, 
				unsigned int endnum, 
				unsigned int fixedwidth,
				unsigned int columnnum,
				vector < double > partition,
				unsigned int SZ,
				vector < symbol_list_ >* v_sym);
/*!
  Overloaded to fill vector of Symbolic_string_ by
  calling constructor internally
*/
void  get_continuous_DataMatrix(unsigned int , 
				unsigned int , 
				unsigned int ,
				unsigned int ,
				vector < double >,
				unsigned int,
				vector < Symbolic_string_ >&);

/*!
  Using multiple data columsn corresponding to multiple 
  sensors seeing the same event
*/
void  get_continuous_DataMatrix(unsigned int begnum, 
				unsigned int endnum, 
				unsigned int fixedwidth,
				vector <unsigned int> columnindices,
				matrix_dbl partition,
				unsigned int SZ,
				vector < Symbolic_string_ >& v_sym);

/*!
  genESeSS input: Overloaded to fill vector of Symbolic_string_ by
  calling constructor internally
*/
void get_continuous_DataMatrix(string filename,
			       vector < double > partition,
			       unsigned int SZ,
			       vector < Symbolic_string_ >&);

/*!
  if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
  as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
  from the \b datfile. As before, \b len is the maximum length of data points read. 
*/
void get_continuous_DataMatrix(string datafile,
			       string data_dir,
			       vector <double> partition,
			       unsigned int len,
			       symbol_list_ &sL);
//---------------------------------------------------------------

/*!
  \b Function for generating random seed. Can call /dev/random or use internal clock via gettimeofday(). The random seed is used to initialize gsl
  random number generators
*/
unsigned long int random_seed();

//---------------------------------------------------------------

/*!
  \b Function for drawing graphviz graph of automaton via boost:graph library
  last two parameters are dotfilename and file type (png,ps etc)
*/

void drawGraph(connx, pitilde, string, string );

//---------------------------------------------------------------

class Set_symbolic_string_
{
  /*!
    Embedding dimension
  */
  unsigned int eDIM;
  /**
     \b TOLERANCE defines the embedding projection error 
     which is ignored to be \c zero
  */
  double TOLERANCE;

  vector <Symbolic_string_> data;
  size_t num_elements;

  /*!
    \b dMap is the storage for the
    pairwuse distance matrix created via
    semantic annihilation
  */
  matrix_dbl dMap;

  /*!
    \b  embeddingCoordinates is the storage for
    embedding coordinates obtained via embedding 
    symmetricized and zero-diagonal version of dMap
  */
  matrix_dbl embeddingCoordinates;

  /*!
    \b embeddingError is the embedding error vs embedding dimension.
    This is the \c maximum projection error if the embedding is cut off at that dimension
  */
  vector <double> embeddingError;

 
  /*!
    \b sae is the self-annihilation error.
    This is the \c diagonal elements of the distance matrix.
    This is only filled if the constructor is called with the SAE_ option
  */
  vector <double> sae;

  /*!
    \b hdiff is the average distance from last HIST streams.
    Useful when stream oindexing correspond to time.
  */
  vector <double> hdiff;

  /*!
    Helper functions for the embedding algorithm
  */
  bool check_for_zero(matrix_dbl*,double*);

  /*!
    Helper functions for the embedding algorithm
  */
  void get_generator(matrix_dbl*,pair <unsigned int, unsigned int> *,double*);


 public:

  /*!
    \b Constructors: default, from \c vector of \c symbols, and 
    from \c Set_symbolic_string_ i.e. the copy constructor
  */
  Set_symbolic_string_();
  Set_symbolic_string_(vector <Symbolic_string_>&);
  Set_symbolic_string_(vector <Symbolic_string_>&, unsigned int);
  Set_symbolic_string_(vector <Symbolic_string_>&, unsigned int, unsigned int NORM_DEPTH);
  Set_symbolic_string_(vector <Symbolic_string_>&, unsigned int, unsigned int NORM_DEPTH, bool SAE_);
  Set_symbolic_string_(vector <Symbolic_string_>&, unsigned int, unsigned int NORM_DEPTH, 
		       unsigned int iO, unsigned int IE, unsigned int jO, unsigned int jE);

  /** Constructor for immediate past comparison
      H_Spec is the history length to be compared against
  */
  Set_symbolic_string_(vector <Symbolic_string_> &S, 
		       unsigned int RPT,
		       unsigned int NORM_DEPTH,bool ONLY_PAST,
		       int H_SPEC );

  /*!
    Constructor for library comparison
  */
  Set_symbolic_string_(vector <Symbolic_string_>& S,
		       vector <Symbolic_string_>& S_LIB,
		       unsigned int, unsigned int DEPTH);
  /*!
    Multi-threaded version. Last parameter specifies the number of 
    calcualtions per thread and last but one specifies thedesired
    parallel threads
  */
  Set_symbolic_string_(vector <Symbolic_string_>&, 
		       unsigned int, unsigned int, unsigned int);
  Set_symbolic_string_(vector <symbol_list_ >&);
  Set_symbolic_string_(const Set_symbolic_string_ &);

  /*!
    Implement insertion or deletion of members
    Overload insertion to work with both 
    Symbolic_string_ and symbol_list_
  */
  void insert_member(const vector <Symbolic_string_>&);
  void insert_member(const symbol_list_&);
  void delete_member(unsigned int);

  /*!
    Regenerate the data structure; 
    compute the distance matrix, embedding and the embedding error
  */
  void regenerate(double);

  matrix_dbl  distance_matrix(void){ return dMap;};

  vector <double>  embedding_error(void){return embeddingError;};
  vector <double>  get_sae(void){return sae;};
  vector <double>  get_hdiff(void){return hdiff;};

  matrix_dbl embedding_coordinates(const unsigned int);


};

//-------------------------------------------------------
/**
   Boost label writer overload to accomodate HTML
   tags in labels (primarily for subscripts in state labels)
*/
template <class Name>
class label_writer_html {
 public:
 label_writer_html(Name _name) : name(_name) {}
  template <class VertexOrEdge>
    void operator()(std::ostream& out, const VertexOrEdge& v) const {
    out << "[label=" << name[v] << "]";
  }
 private:
  Name name;
};
template <class Name>
inline label_writer_html<Name>
make_label_writer_html(Name n) {
  return label_writer_html<Name>(n);
}

//-------------------------------------------------------
/**
   Boost label writer overload to accomodate other prperties
   tags in labels (primarily for subscripts in state labels)
*/


template <class Name>
class label_writer_noquotes {
 public:
 label_writer_noquotes(Name _name) : name(_name) {}
  template <class VertexOrEdge>
    void operator()(std::ostream& out, const VertexOrEdge& v) const {
    out << "[" << name[v] << "]";
  }
 private:
  Name name;
};
template <class Name>
inline label_writer_noquotes<Name>
make_label_writer_noquotes(Name n) {
  return label_writer_noquotes<Name>(n);
}

//---------------------------------------------------------------
/*!
  \b Sippl_embed is the class for carrying out embedding of a distance
  matrix. This is provided seprately, so as to carry out embeddings 
  withou first carrying out smashing as done within the constructor of 
  \b Set_symbolic_string_
*/
class Sippl_embed
{

 private:

  vector <double> embeddingError;
  double TOLERANCE;
  unsigned int eDIM;
  bool check_for_zero(matrix_dbl*,double*);
  void get_generator(matrix_dbl*,pair <unsigned int, unsigned int> *,double*);
  /*!
    \b  embeddingCoordinates is the storage for
    embedding coordinates obtained via embedding 
    symmetricized and zero-diagonal version of dMap
  */
  matrix_dbl embeddingCoordinates;

 public:

  Sippl_embed();
  Sippl_embed(matrix_dbl*);
  Sippl_embed(matrix_dbl*,double);

  matrix_dbl embedding_coordinates(const unsigned int);
  vector <double> &  embedding_error(void){return embeddingError;};

};



//---------------------------------------------------------------
//    genESeSS declarations to follow 
//---------------------------------------------------------------


/*!
  \b \f$ \epsilon\f$-resolution comparison class needed for
  choosing resolution of state merging in the genESeSS algorithm
*/
class eps_compare_ { 

 public:
  eps_compare_(){epsilon=0.1;};		
  eps_compare_(double s_){epsilon=s_;};		
  bool operator()(const vector <double> &lhs,const vector <double> &rhs) const 
  { 
    double S=0.0;
    for(unsigned int i=0; i < lhs.size();++i)
      S+=fabs(lhs[i]-rhs[i]);
    return (S>epsilon) && (lhs < rhs); 
  }

 private:
  double epsilon;
};

//------------------------------------------------

//------------------------------------------------

/*!
  Compute entropy of probability vector (in bits)
*/
double dist_entropy(vector <double> p); 

/*!
  Normalize integer vector to probability vector 
*/
vector <double> mk_stochastic(map <symbol, unsigned int > &);

/*!
  Normalize integer matrix to row stochastic matrix
*/
void mk_stochastic(phi_data_type_ &,stoch_phi_data_type_ &);

/*!
  Compute square of L2 norm of the difference of integer vectors
  Note that we do not take square root for efficiency
*/
double l2dev(map <symbol, unsigned int > &, map <symbol, unsigned int > &);

/*!
  Compute  Linf norm of the difference of  vectors
*/
double vec_distance(vector <double> v0, vector <double> v1);

/*!
  Determine if a map id all zero
  and hence cannot be normalized
*/
bool zero(map <symbol, unsigned int > &);

//------------------------------------------------

//------------------------------------

class gParam {

 public:

  double MIN_DEV_SYNC_SEARCH_;
  unsigned int LOOP_COUNT_SYNC_SEARCH_;
  unsigned int  SYN_PREF_LENGTH_;
  double EPS_MIN_;
  double EPS_MAX_;
  double EPS_STEP_;
  unsigned int ANN_TEST_LEN_;
  bool USE_EPS_RANGE;
  bool COL_PRINT;
  unsigned int  EVALUATION_DEPTH;
  bool EXTREMELY_SHORT_;
  int NUM_EACH;
  int   MIN_STATE_SIZE;
  bool USE_GAMMA;
  gParam();
  gParam(string configfile,unsigned int EVALDEPTH=8);

};
//------------------------------------


/*!
  Class genESeSS that houses the 
  data structures and the algorithms for
  generating a PFSA from symbolic data stream
*/
class genESeSS {

 protected:

  symbol_list_ data;
  phi_data_type_ Phi;
  pin_data_type_ Datapin;
  stoch_phi_data_type_ sPhi;
  unsigned int alphabet;
  symbol_list_ omega_syn;
  vector <double> eps_vec;
  vector<connx> aut_vec;
  vector <pitilde> pit_vec;

  bool NO_STRONG_COMPONENTS_;
  PFSA G_mc_;

  /**
     genESeSS parameters
  */
  double MIN_DEV_SYNC_SEARCH_;
  unsigned int LOOP_COUNT_SYNC_SEARCH_;
  unsigned int  SYN_PREF_LENGTH_;
  double EPS_MIN_;
  double EPS_MAX_;
  double EPS_STEP_;
  unsigned int ANN_TEST_LEN_;
  bool COL_PRINT;
  unsigned int  EVALUATION_DEPTH;

 public:

  /*! Constructor */
  genESeSS(){};

  /*! Constructor */
  genESeSS(symbol_list_&, unsigned int);
  genESeSS(Symbolic_string_ , gParam);

  virtual void getPhi(symbol_list_ );
  virtual void getPit();

  // symbol_list_ getSync(symbol_list_);
  symbol_list_ getSync();

  void getAut();
  /*! To ensure the model is strongly connected */
  vector < connx > getComponent(connx );

  /*! Accessor function for symbolic derivative */
  vector <double>  phi(symbol_list_ lambda) {getPhi(lambda);
    return mk_stochastic(Phi[lambda]);};

  /*! Accessor function for the matrix of computed symbolic derivative */
  phi_data_type_ &phi(){return Phi;};

  /*! Accessor function for inferred machine  */
  PFSA &get_mc(){return G_mc_;};

  /*! Utility function for incrementing \c symbol_list_ */
  symbol_list_ & inc(symbol_list_ &);
  
  /*! Accessor function for inferred synchronizing string  */
  symbol_list_ &sync_string(){return omega_syn; };

  /*! Optimal \f$ \epsilon \fS inferred using information annihilation  */
  double eps;

  /*! Deviation from flat white noise of the  stream
    obtained via annihilating the one obtained from inverting 
    a stream generated from the inferred PFSA
    against the input symbolic stream  */
  double annihilation_error;


  /*! ONLY FOR X: How well can we predict individual symbols, 
    sum of row-wise "entropies" of \b pit. If local error  
    is zero, then we can predict next symbol prefectly, which means 
    each row in pit of the xPFSA has only zeroes and ones; 
    which in turn means that the sum of the rowwise entropies is zero  */
  double local_error;


  /*! ONLY FOR X: local_error times annihilation_error  */
  double combined_error;


  /*!
    \b Operator << overloaded for  pretty printing of inferred PFSA
  */
  friend ostream& operator<< (ostream &, genESeSS &);


  /*!
    Utility function to print colored strings (*NIX terminals only)
  */
  string _color(string str);
  string set_color(int forecolor, int backcolor)
  { 
    string str="";
    if(COL_PRINT)
      {
	char buff[1024]; 
	sprintf(buff,"%c[%d;%d;%dm",27,_semantic_RESET,forecolor,backcolor);
	str+=buff;
      }
    return str;
  };
  string reset_color()
  { char buff[1024]; string str; sprintf(buff,"%c[%dm",27,_semantic_RESET);return str+buff;};

};


//------------------------------------

/*!
  Class for XgenESeSS. It is derived from the base class \b genESeSS
  Note the use of virtual functions which allow 
  handling cross-dependencies between two input streams
*/
class genESeSS_multistream : public genESeSS
{
 protected:
  vector <symbol_list_ > data;
  vector < pin_data_type_> Datapin;

 public:
  genESeSS_multistream(vector<symbol_list_> &, unsigned int);
  genESeSS_multistream(vector<Symbolic_string_ > &, gParam);
  genESeSS_multistream(vector< symbol_list_> &, gParam);

  void getPhi(symbol_list_ );
  void getPit();
};

//------------------------------------

/*!
  Class for handling multi-line input
  data in genESeSS. It is derived from the base class \b genESeSS
  Note the use of virtual functions which allow 
  defining the multi-line processing via overriding 
  two functions
*/
class genESeSS_x : public genESeSS
{
 protected:
  symbol_list_  data_A;
  symbol_list_  data_B;
  unsigned int alphabet_A;
  unsigned int alphabet_B;
  size_t data_size;
  bool USE_GAMMA;
  double Xgamma;
  
 public:
  genESeSS_x(Symbolic_string_  , Symbolic_string_  , gParam);

  void getPhi(symbol_list_ );
  void getPit();
  
  double gamma(){return Xgamma;};
  Symbolic_string_  run_mc(connx &aut, pitilde &pit);

};

//------------------------------------
//-----------------
//---------------------------------
/*!
  Class for a simple non-blocking stream server
  meant to implement a polling \b data_reader
*/
class server_
{
  boost::mutex io_mutex;

  list <symbol> data_;
  unsigned int window;
  boost::thread *T_;
  unsigned int MAX_SIZE_;

 public:

  int newsockfd;
  int sockfd;
  socklen_t clilen;
  struct sockaddr_in serv_addr, cli_addr;

  /** Constructor   **/
  server_(int portno, int);

  /** Adds to the internal polling data store.
      Also, makes sure that the data store is 
      maintained within the maximum size /b MAX_SIZE_  **/
  void add_data(list <symbol>);

  /** Accessor function for internal data **/
  const list <symbol> &data();

  /** Clears the internal store **/
  void clear(); 
};
//---------------------------------
//---------------------------------
/** Utility for returning error  **/
void error(const char *msg);

/** Polling function used in server thread **/
void data_poll_(server_*);

//---------------------------------
//---------------------------------
/*!
  Class for simple stream client 
  to send data to the simple stream server
*/
class client_
{
  int sockfd;
  struct sockaddr_in serv_addr, cli_addr;
  struct hostent *server;

 public:

  client_(int portno, const char*);
  //  call as: client_ C(portno, hostaddress);
  //  where eg hostaddress=argv[2]
  void send(string);
  void close_client(void){close(sockfd);};
};
//---------------------------------
//---------------------------------
/*!
  Class for reading data files in
  various formats, and with both 
  continuous values (converted internally to symbols
  via a specified partition), and also
  direct symbol streams
*/
class data_reader
{

 private:
  bool SERVER_MODE;
  server_ *Server;

  vector <symbol_list_> data;
  map <string,int> keys_;

 public:

  /*!
    Read symbols from the file \b sdata_file
    assuming each line is a new data stream, i.e., 
    file is read "across". To read file "up" in columns,
    use different constructor which provides 
    for \b data_dir. Reads only upto \b SZ
  */
  data_reader(string sdata_file,
	      unsigned int SZ);

  /*!
    Read symbols from the file \b sdata_file
    assuming each line is a new data stream, i.e., 
    file is read "across". To read file "up" in columns,
    use different constructor which provides 
    for \b data_dir. 
  */
  data_reader(string sdata_file);


  /*!
    if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
    as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
    from the \b datfile. \b SZ is the maximum length of data points read. 
  */

  data_reader(string sdata_file,
	      string data_dir,
	      unsigned int SZ,
	      bool KEY_=false);

  /*!
    if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
    as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
    from the \b datfile. 
  */
  data_reader(string sdata_file,
	      string data_dir);


  /*!
    Read continuous values from the file \b cdata_file
    assuming each line is a new data stream, i.e., 
    file is read "across". To read file "up" in columns,
    use different constructor which provides 
    for \b data_dir. Reads only upto \b SZ
  */
  data_reader(string cdata_file, 
	      vector < double > partition,
	      unsigned int SZ);

  /*!
    Read continuous values from the file \b cdata_file
    assuming each line is a new data stream, i.e., 
    file is read "across". To read file "up" in columns,
    use different constructor which provides 
    for \b data_dir
  */
  data_reader(string cdata_file, 
	      vector < double > partition);
  /*!
    if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
    as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
    from the \b datfile. As before, \b len is the maximum length of data points read. 
  */

  data_reader(string cdata_file,
	      string data_dir,
	      vector <double> partition,
	      unsigned int SZ,
	      bool KEY_=false,
	      bool DERIVATIVE=false);


  /*!
    Datareader with the possibility of variable quantization.
    if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
    as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
    from the \b datfile. As before, \b len is the maximum length of data points read. 
  */

  data_reader(string cdata_file,
	      string data_dir,
	      vector< vector <double> > partition,
	      unsigned int SZ,
	      bool KEY_=false);




  /*!
    if \c string \b data_dir is set to  "across", then the \c symbol_list_ is read 
    as a row in the \b datafile. Else, the \c symbol_list_ is read as a column 
    from the \b datfile. As before, \b len is the maximum length of data points read. \c bool \b KEY_ is a flag to read leading "keys" in each data line
  */

  data_reader(string cdata_file,
	      string data_dir,
	      vector <double> partition);

  /*!
    Server mode to poll data on a thread
  */
  data_reader(int portnumber, 
	      int len);


  /*!
    Accessor function to get first symbol list
  */
  symbol_list_  getlist();

  /*!
    Accessor function to get vector of  symbol lists
  */
  vector < symbol_list_ >  getlist_vector(){ return data; };

  /*!
    Accessor function to get \c Symbolic_string constructed from
    the first symbol list
  */
  Symbolic_string_ getsymbolic_string();

  /*!
    Accessor function to get \c Symbolic_string \c vector constructed from
    the full symbol list
  */
  vector < Symbolic_string_ > getsymbolic_string_vector();
  map <string,int> keys();


};

//------------------------------------

class parse_   
{        
 protected:  
  string input_;
  string output_;   
                     
  const set<string> Operator;
  const map<string,unsigned int> arity;
  
  set<string> varnames_;
  
 public:        
  parse_(char*,set<string>,map<string,unsigned int>);   
  parse_(string,set<string>,map<string,unsigned int>);   
                
  string get();
  vector<string> get(string& expr,
		     string placeholder_=" __REPLACE__ ");
  
  string spc(char*);    
  string spc(string);          
  set<string>& varnames();
  
  void ReplaceStringInPlace(string& subject,  
                            const string& search,   
                            const string& replace); 
                                
  friend ostream& operator<< (ostream &, parse_ &);    
};                       
//--------------------------------   
//--------------------------------
/*!
  \b spinner in the dats structure used by the spin python class.
  They are actually XPFSAs, along with exyra info on \c gamma , \c delay,
  \c distance, and \c src , \c tgt info.
 */
class spinner: public PFSA
{
  double gamma_;
  unsigned int delay_;
  string src_,tgt_;
  double distance_;
  
public:
  spinner(pitilde p,
	  connx a,
	  double g,
	  unsigned int d,
	  string s,
	  string t,
	  double distance=-1.0):PFSA(p,a)
  {
    gamma_=g;
    delay_=d;
    src_=s;
    tgt_=t;
    distance_=distance;
  };
  
  void gamma(double g){gamma_=g;};
  void delay(unsigned int d){delay_=d;};
  void src(string s){src_=s;};
  void tgt(string t){tgt_=t;};
  void distance(double dist){distance_=dist;};

  double gamma(){return gamma_;};
  double distance(){return distance_;};
  unsigned int delay(){return delay_;};
  string src(){return src_;};
  string tgt(){return tgt_;};

};
//------------------------------------------




//------------------------------------
/**
   Utility function to reflect symbolic lists
*/
symbol_list_ reflect(symbol_list_  S, unsigned int alphabet);


/**
   pretty print utilities
*/
ostream& operator<< (ostream &, matrix_dbl & );
//ostream& operator<< (ostream &, vector <double> & );
ostream& operator << (ostream &out, parse_& s);

//------------------------------------
ostream& operator << (ostream &, connx &);
//------------------------------------
ostream& operator << (ostream &, symbol_list_ &);
//------------------------------------
ostream& operator << (ostream &, vector<double>);
//------------------------------------
ostream& operator << (ostream &,  
		      map < symbol, unsigned int > &);
//------------------------------------
ostream& operator << (ostream &, phi_data_type_ &);
//------------------------------------
ostream& operator << (ostream &, stoch_phi_data_type_ &);
//------------------------------------
 
/*!
  distance matrix for a seq vector given a vector of PFSA
*/
matrix_dbl llk_distance(vector<symbol_list_>& s,
			vector<PFSA>& G,
			unsigned int alphabet=0);

matrix_dbl llk_distance(vector<symbol_list_>& s);




//------------------------------------
namespace SCC_INNER_PROD__
{

  double mult(const vector<double>&,
	      const vector<double>&);

};
//------------------------------------

namespace SCC_UTIL__
{
  /*!
    Utility function to produce progress bar
   */
  void print_status(float progress);
  
  /*!
    Utility function to print message. If the string ERROR occurs in the message
    then execution is aborted 
   */
  void _MESSAGE(string,
		const char *caller);

  /*!
    Utility function to remove substring \c const \c string
    from a given \c string returns string
  */
  string removeSubstrs__(string&);

  /*!
    Utility function to remove substring \c const \c string
    from a given \c string
  */
  bool removeSubstrs(string&, 
		     const string& );
  /*!
    Utility function to replace all occurrences of 
    substring \c const \c string \b search 
    with  \c const \c string \b replace
    from a given \c string subject 
  */
  void ReplaceStringInPlace(string& subject, 
			    const string& search,
			    const string& replace);
  /*!
    Utility function to remove leading and trailing whitespace 
    substring \c consist \c string \b whitespace   from a given \c string.
    The white space can be given to be any string, default is \b whitespace
  */
  string trim(const string& str,
	      const string& whitespace = " ");

  /*! Utility to check if input is double or string */
  bool isOnlyDouble(const char* str);

  /*! Utility to read PFSA from file */  
  PFSA read_mc(string filename, string TYPE__);


  /*! PFSA to generate flat white noise */
  PFSA FWN(unsigned int alph_);  

  /*! utility to map vector of ints to map_sym_state */
  map_sym_state mapto_symstate(vector<int> v);

  /*! read XPFSAs from json file */
  vector <spinner> read_json(string jsonfile);

  /*!
    Merge states that can be merged. Used by ~
  */
  bool merge_states(PFSA& G, double eps=0.0001);


  /*!
    generate standardized machines of type M, S or T 
   */
  PFSA generate_mc(unsigned int alphabet,
		   unsigned int numstates,
		   string MC_TYPE="M");
};

#endif
