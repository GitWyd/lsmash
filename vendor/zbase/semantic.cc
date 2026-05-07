//      semantic.cc
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

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <utility>
#include <sys/time.h>
#include <unistd.h>
#include <set>
//#include <boost/thread/thread.hpp> 
//#include <boost/thread/mutex.hpp>
//#include <boost/bind.hpp>
#include <omp.h>

#include "semantic.h"

using namespace std;
using namespace boost;

//boost::mutex io_mutex;

const char sep=' ';
volatile unsigned long int global_seed = 1;

//bool GLOBAL_VAR_DATA_LEN_SET=false;
//int GLOBAL_VAR_DATA_LEN=-1;


//--------------------------------------------------------------------------

/**
   Necessary for printing of \b symbol which is of type unsigned char
   /b HexCharStruct is the structure
*/
template <class T>
struct HexCharStruct
{
  T c;
  HexCharStruct(T _c) : c(_c) { }
};
/**
   Necessary for printing of \b symbol which is of type unsigned char
   Overloading the << operator
*/
template <class T>
inline ostream& operator<<(ostream& o, const HexCharStruct<T>& hs)
{
  return (o << std::hex << static_cast<int>(hs.c));
}
/**
   Necessary for printing of \b symbol which is of type unsigned char
*/
template <class T>
inline HexCharStruct <T> hex(T _c){ return HexCharStruct<T>(_c);}

//--------------------------------------------------------------------------
ostream& operator << (ostream &out, PFSA &G)
{
  //char sep = '|';
  for (symbol_list_::iterator itr=G.data.begin(); 
       itr != G.data.end(); 
       ++itr)
    out << hex<symbol>(*itr) << sep;
  return out;
}
//--------------------------------------------------------------------------
ostream& operator << (ostream &out, Symbolic_string_ s)
{
  //char sep = '|';
  for (symbol_list_::iterator itr=s.data.begin(); 
       itr != s.data.end(); 
       ++itr)
    out << hex<symbol>(*itr) << sep;
  return out;
}
//--------------------------------------------------------------------------
ostream& operator << (ostream &out, matrix_dbl &s)
{
  for(matrix_dbl::iterator itr=s.begin();
      itr!=s.end();
      ++itr)
    {
      for(map<unsigned int,double>::iterator itr1=itr->second.begin();
	  itr1!=itr->second.end();
	  ++itr1)
	out << itr1->second << " ";
      out << endl;
    }
/*  for (unsigned int i=0; i<s.size(); i++)
    {    
      for (unsigned int j=0; j<s[0].size(); j++)
	out << s[i][j] << " ";
      out << endl;
    }
*/
   return out;
}
//--------------------------------------------------------------------------
ostream& operator << (ostream &out, vector <double> &s)
{
  for (unsigned int i=0; i<s.size(); i++)
    out << s[i] << endl;
  out << endl;
  return out;
}

//--------------------------------------------------------------------------
double PFSA::log_likelihood(const symbol_list_& s)
{
  if(s.empty())
    return 0.0;
  
  double llk=0.0;
  const unsigned int numstates=get_aut().size();

  map<symbol,vector<double> > map_pitcol;
  for(unsigned int st=0;st<numstates;++st)
    for(symbol i(0); i < get_aut()[0].size();++i)
      map_pitcol[i].push_back(get_pit()[st][i]);

  vector<double> curr_state(get_Stationary());

  for(unsigned int i=0;i<s.size();++i)
    {
      double pr=0.0;
      for(unsigned int st=0;st<numstates;++st)
	pr+=map_pitcol[s[i]][st]*curr_state[st];

      llk+=log2(pr);

      vector <double> state_vec_tmp(curr_state);
      for (unsigned int k=0; k < numstates; k++)
	{
	  double V=0.0000001;
	  for (unsigned int j=0; j < numstates; j++)
	    V+=get_Gamma()[s[i]][j][k]*state_vec_tmp[j];
	  curr_state[k]=V;
	}
      
      double S=0.0;
      for (unsigned int i=0;i<numstates;i++)
	S+=curr_state[i];

      if(S>0.0)
	S=1/S;
      
      for (unsigned int i=0;i<numstates;i++)
	curr_state[i] *= S;	
    }

  return -llk/s.size();
};

//--------------------------------------------------------------------------
vector<double> PFSA::log_likelihood(const vector<symbol_list_> & s_) 
{
  if(s_.empty())
    return vector<double> (s_.size(),0.0);
  
  vector<double> llk(s_.size(),0.0);
  const unsigned int numstates=get_aut().size();

  map<symbol,map<unsigned int,double> > map_pitcol;
  for(unsigned int st=0;st<numstates;++st)
    for(symbol i(0); i < get_aut()[0].size();++i)
      map_pitcol[i][st]=get_pit()[st][i];


  
  //map<symbol,vector<double> > map_pitcol;
  /*  for(unsigned int st=0;st<numstates;++st)
    for(symbol i(0); i < get_aut()[0].size();++i)
      map_pitcol[i].push_back(get_pit()[st][i]);
  */
  vector<double> curr_state_0(get_Stationary());
  
  //#pragma omp parallel for
  for(unsigned int seq=0;seq<s_.size();++seq)
    {
      vector<double> curr_state=curr_state_0;
      symbol_list_ s=s_[seq];
      for(unsigned int i=0;i<s.size();++i)
	{
	  double pr=0.0;
	  for(unsigned int st=0;st<numstates;++st)
	    {
	      //	      cout << "pr " << pr << " ssize " << s.size() << " i " << i << endl;
	      //cout << i << " " << s[i] << endl;
	      if (map_pitcol.find(s[i])!=map_pitcol.end())
		pr+=map_pitcol[s[i]][st]*curr_state[st];
	      
	    }

	  llk[seq]+=log2(pr);

	  vector <double> state_vec_tmp(curr_state);
	  for (unsigned int k=0; k < numstates; k++)
	    {
	      double V=0.0000001;
	      for (unsigned int j=0; j < numstates; j++)
		if (s[i]<map_pitcol.size())
		  V+=get_Gamma()[s[i]][j][k]*state_vec_tmp[j];
	      curr_state[k]=V;
	    }
      
	  double S=0.0;
	  for (unsigned int i=0;i<numstates;i++)
	    S+=curr_state[i];

	  if(S>0.0)
	    S=1/S;
      
	  for (unsigned int i=0;i<numstates;i++)
	    curr_state[i] *= S;	
	}
      llk[seq]=-llk[seq]/s.size();
    }
  return llk;


};

//--------------------------------------------------------------------------
PFSA& PFSA::operator* (double alpha)
{
  //lock_guard<mutex> guard(mtx_);

  PFSA *tmp;
  tmp = new PFSA;

  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());

  tmp->pit=pit;
  tmp->Xpit=Xpit;

  map < state,  vector < double > >::iterator itr=tmp->pit.begin();
  size_t alph = (itr->second).size();
  for ( map < state,  vector < double > >::iterator itr_tmp=tmp->pit.begin();
	itr_tmp!=tmp->pit.end();++itr_tmp)
    {
      double sum=0.0;
      for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
	  itr_row!=(itr_tmp->second).end();++itr_row)
	{
	  if (*itr_row > 0.0)
	    *itr_row=pow(*itr_row,alpha);
	  sum+=*itr_row;
	}
      if (sum>0)
	for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
	    itr_row!=(itr_tmp->second).end();++itr_row)
	  (*itr_row)=(*itr_row)/sum;
    }  
  tmp->aut=aut; // same structure
  srand ( time(NULL) );
  if(aut.size() == 1)
    tmp->curr_state=0;
  else
    tmp->curr_state=rand()%(pit.size()-1);
  //  tmp->curr_state=curr_state;
  tmp->alphabet = aut[0].size();
  tmp->NUM_STATE=aut.size();

  size_t len=data.size();
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(tmp->pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(tmp->aut[curr_state][sym]!=-1)
	{
	  tmp->data.push_back(sym);
	  tmp->curr_state=tmp->aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);

  return *tmp;

}
//--------------------------------------------------------------------------
PFSA& PFSA::operator! ()
{
  //lock_guard<mutex> guard(mtx_);
  PFSA *tmp;
  tmp = new PFSA;

  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());

  tmp->pit=pit;
  tmp->Xpit=Xpit;
  map < state,  vector < double > >::iterator itr=tmp->pit.begin();
  size_t alph = (itr->second).size();
  for ( map < state,  vector < double > >::iterator itr_tmp=tmp->pit.begin();
	itr_tmp!=tmp->pit.end();++itr_tmp)
    {
      double sum=0.0;
      for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
	  itr_row!=(itr_tmp->second).end();++itr_row)
	{
	  if (*itr_row > 0.0)
	    *itr_row=1/(*itr_row);
	  else
	    *itr_row=1.0;
	  sum+=*itr_row;
	}
      if (sum>0)
	for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
	    itr_row!=(itr_tmp->second).end();++itr_row)
	  (*itr_row)=(*itr_row)/sum;
    }  
  
  tmp->aut=aut; // same structure
  tmp->alphabet = aut[0].size();
  tmp->NUM_STATE=aut.size();

  srand ( time(NULL) );
  if (pit.size() > 1)
    tmp->curr_state=rand()%(pit.size()-1);
  else
    tmp->curr_state=0;

  size_t len=data.size();
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(tmp->pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(tmp->aut[curr_state][sym]!=-1)
	{
	  tmp->data.push_back(sym);
	  tmp->curr_state=tmp->aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);

  return *tmp;
}
//--------------------------------------------------------------------------
connx& PFSA::get_aut() 
{
  return aut;
}
//--------------------------------------------------------------------------

pitilde& PFSA::get_pit() 
{
  return pit;
}
//--------------------------------------------------------------------------

pitilde& PFSA::get_Xpit() 
{
  return Xpit;
}
//--------------------------------------------------------------------------
void PFSA::set_aut(connx &aut1) 
{
  aut = aut1;
}
//--------------------------------------------------------------------------

void PFSA::set_pit(pitilde &pit1) 
{
  pit = pit1;
}
//--------------------------------------------------------------------------

void PFSA::set_Xpit(pitilde &pit1) 
{
  Xpit = pit1;
}

//--------------------------------------------------------------------------

PFSA& PFSA::operator+ (PFSA& G)
{

  PFSA thisGX(*this || G);
  PFSA GthisX_(G || *this);
  PFSA GthisX = GthisX_.transpose_name(this->get_aut().size());
  
  connx aut1(thisGX.get_aut());

  pitilde pitTG(thisGX.get_pit());
  pitilde pitGT(GthisX.get_pit());

  pitilde pit1;

  for(state q=0; q<(int)aut1.size();++q)
    {
      vector <double> pitrow1 = pitTG[q];
      vector <double> pitrow2 = pitGT[q];
      vector <double> pitrow(pitrow1.size(),0.0);

      for(unsigned int i=0;i<pitrow1.size();++i)
	pitrow[i] = pitrow1[i]*pitrow2[i];

      double S=0.0;
      for(unsigned int i=0;i<pitrow.size();++i)
	S+=pitrow[i];

      if (S>0.0)
      for(unsigned int i=0;i<pitrow.size();++i)
	pitrow[i]/=S;

      pit1[q] = pitrow;
    }

  PFSA P(pit1,aut1);
  while(SCC_UTIL__::merge_states(P)){};   

  PFSA* R;
  R= new PFSA(P.get_pit(),P.get_aut());
  
  return *R;
}
//--------------------------------------------------------------------------  //SCC_UTIL__::merge_states(*R);


PFSA& PFSA::operator| (PFSA& G)
{
  connx aut1(G.get_aut());
  pitilde pit1(G.get_pit());

  if (pit1.empty())
    {
      if (!aut1.empty())
	{
	  vector <double> U(aut[0].size(),1/(aut[0].size()+0.0));
	  for (state q=0; q < (int) aut.size();++q)
	    pit1[q]=U;
	}
      else
	return *this;
    }

  map < state, map < state , state > > state_index;
  map < symbol, map < symbol, symbol > > symbol_index;

  unsigned int count=0;
  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int)aut1.size(); ++q1)
      state_index[q][q1]=count++;

  symbol symc(0);
  for (symbol  s(0); s < aut[0].size(); ++s)
    for (symbol s1(0); s1 < aut1[0].size(); ++s1)
      symbol_index[s][s1]=symc++;

  connx autX;
  pitilde pitX;

  for (state q=0; q <(int) aut.size(); ++q)
    for (state q1=0; q1 < (int) aut1.size(); ++q1)
      for (symbol  s(0); s < aut[0].size(); ++s)
	for (symbol s1(0); s1 < aut1[0].size(); ++s1)
	  autX[state_index[q][q1]][symbol_index[s][s1]] = state_index[aut[q][s]][aut1[q1][s1]];

  vector <double> pitrow(autX[0].size(),0.0);
  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int)  aut1.size(); ++q1)
      pitX[state_index[q][q1]] = pitrow;

  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int) aut1.size(); ++q1)
      for (symbol  s(0); s < aut[0].size(); ++s)
	for (symbol s1(0); s1 < aut1[0].size(); ++s1)
	  pitX[state_index[q][q1]][symbol_index[s][s1]] = pit[q][s]*pit1[q1][s1];

  PFSA *tmp;
  tmp = new PFSA;
  tmp->aut=autX; // same structure
  tmp->pit = pitX;
  tmp->alphabet = autX[0].size();
  tmp->NUM_STATE=autX.size();
  srand ( time(NULL) );
  if (pit.size() > 1)
    tmp->curr_state=rand()%(pit.size()-1);
  else
    tmp->curr_state=0;

  return *tmp;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
PFSA& PFSA::transpose_name(unsigned int colnum)
{
  unsigned int numstates=aut.size();
  if (numstates<1)
    return *this;

  
  unsigned int alphabet=aut[0].size();
  map<state,state> rename_map;
  rename_map[-1]=-1;
  
  unsigned int count=0;

  for(unsigned int col=0;col<colnum;++col)
    for(state q=col;q<(int)numstates;q+=colnum)
      rename_map[q]=count++;

  pitilde pit1;
  for (state q=0;q<(int)numstates;++q)
    pit1[rename_map[q]]=pit[q];

  connx aut1;
  for(state q=0;q<(int)numstates;++q)
    for(symbol s(0);s<alphabet;++s)
      aut1[rename_map[q]][s]=rename_map[aut[q][s]];
  
  PFSA *Gtmp;
  PFSA tmp(pit1,aut1);
  Gtmp = new PFSA (tmp); //find strong component needs tobe ~tmp CHECK
  //Gtmp = new PFSA (tmp); //find strong component needs tobe ~tmp CHECK
  Gtmp->alphabet = aut1[0].size();
  return *Gtmp;


};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------


PFSA& PFSA::operator|| (PFSA& G)
{
  connx aut1(G.get_aut());
  if (aut1.empty())
    return *this;

  map < state, map < state , state > > state_index;

  unsigned int count=0;
  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int)aut1.size(); ++q1)
      state_index[q][q1]=count++;

  connx autX;
  pitilde pitX;
  pitilde Xpit;

  for (state q=0; q <(int) aut.size(); ++q)
    for (state q1=0; q1 < (int) aut1.size(); ++q1)
      for (symbol  s(0); s < aut[0].size(); ++s)
	autX[state_index[q][q1]][s] = state_index[aut[q][s]][aut1[q1][s]];

  vector <double> pitrow(pit[0].size(),0.0);
  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int)  aut1.size(); ++q1)
      pitX[state_index[q][q1]] = pitrow;

  for (state q=0; q < (int) aut.size(); ++q)
    for (state q1=0; q1 < (int) aut1.size(); ++q1)
      for (symbol  s(0); s < pit[0].size();s++)
	pitX[state_index[q][q1]][s] = pit[q][s];


  PFSA tmp(pitX,autX);
  if (!G.Xpit.empty())
    {
      vector <double> pitrow(G.Xpit[0].size(),0.0);
      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int)  aut1.size(); ++q1)
	  Xpit[state_index[q][q1]] = pitrow;

      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int) aut1.size(); ++q1)
	  for (symbol  s(0); s < G.Xpit[0].size(); ++s)
	    Xpit[state_index[q][q1]][s] = G.Xpit[q1][s];

      tmp.set_Xpit(Xpit);
    }

  PFSA *Gtmp;
  Gtmp = new PFSA (~tmp); //find strong component needs tobe ~tmp CHECK
  //Gtmp = new PFSA (tmp); //find strong component needs tobe ~tmp CHECK
  Gtmp->alphabet = autX[0].size();
  return *Gtmp;
}
//--------------------------------------------------------------------------
PFSA&  PFSA::operator >> (PFSA &G)
{
  unsigned int ITER=10000;
  PFSA *Gtmp;
  connx aut1(G.get_aut());
  if (!aut1.empty())
    {
      map < state, map < state , state > > state_index;

      unsigned int count=0;
      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int)aut1.size(); ++q1)
	  state_index[q][q1]=count++;

      connx autX;
      pitilde pitX;

      for (state q=0; q <(int) aut.size(); ++q)
	for (state q1=0; q1 < (int) aut1.size(); ++q1)
	  for (symbol  s(0); s < aut[0].size(); ++s)
	    autX[state_index[q][q1]][s] = state_index[aut[q][s]][aut1[q1][s]];

      vector <double> pitrow(pit[0].size(),0.0);
      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int)  aut1.size(); ++q1)
	  pitX[state_index[q][q1]] = pitrow;

      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int) aut1.size(); ++q1)
	  for (symbol  s(0); s < pit[0].size(); ++s)
	    pitX[state_index[q][q1]][s] = pit[q][s];

      PFSA Gtmp_(pitX,autX);
      Gtmp_.gen_Stationary(ITER);
      vector <double> stProb(Gtmp_.get_Stationary());

      pitilde pit1;

      for (state q1=0; q1 < (int) aut1.size(); ++q1)
	{
	  pit1[q1] = vector <double> (aut1[0].size(),0.0);
	  double pitrowsum=0.0;
	  for (symbol  s(0); s < aut1[0].size(); ++s)
	    {
	      double stprob_=0.0;
	      for (state q=0; q < (int) aut.size(); ++q)
		stprob_ += stProb[state_index[q1][q]]*pitX[state_index[q1][q]][s];
	      pit1[q1][s] = stprob_;
	      pitrowsum += stprob_;
	    }
	  if (pitrowsum > 0)
	    for (symbol  s(0); s < aut1[0].size(); ++s)
	      pit1[q1][s] /= pitrowsum;
	}

      Gtmp = new PFSA (G);
      Gtmp->set_pit(pit1);
    }
  else 
    Gtmp = new PFSA ();
  return *Gtmp;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
PFSA&  PFSA::sync_proj(PFSA &G, unsigned int ITER)
{
  PFSA *Gtmp;
  connx aut1(G.get_aut());
  if (!aut1.empty())
    {
      map < state, map < state , state > > state_index;

      unsigned int count=0;
      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int)aut1.size(); ++q1)
	  state_index[q][q1]=count++;

      connx autX;
      pitilde pitX;

      for (state q=0; q <(int) aut.size(); ++q)
	for (state q1=0; q1 < (int) aut1.size(); ++q1)
	  for (symbol  s(0); s < aut[0].size(); ++s)
	    autX[state_index[q][q1]][s] = state_index[aut[q][s]][aut1[q1][s]];

      vector <double> pitrow(pit[0].size(),0.0);
      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int)  aut1.size(); ++q1)
	  pitX[state_index[q][q1]] = pitrow;

      for (state q=0; q < (int) aut.size(); ++q)
	for (state q1=0; q1 < (int) aut1.size(); ++q1)
	  for (symbol  s(0); s < pit[0].size(); ++s)
	    pitX[state_index[q][q1]][s] = pit[q][s];

      PFSA Gtmp_(pitX,autX);
      Gtmp_.gen_Stationary(ITER);
      vector <double> stProb(Gtmp_.get_Stationary());

      pitilde pit1;

      for (state q1=0; q1 < (int) aut1.size(); ++q1)
	{
	  pit1[q1] = vector <double> (aut1[0].size(),0.0);
	  double pitrowsum=0.0;
	  for (symbol  s(0); s < aut1[0].size(); ++s)
	    {
	      double stprob_=0.0;
	      for (state q=0; q < (int) aut.size(); ++q)
		stprob_ += stProb[state_index[q1][q]]*pitX[state_index[q1][q]][s];
	      pit1[q1][s] = stprob_;
	      pitrowsum += stprob_;
	    }
	  if (pitrowsum > 0)
	    for (symbol  s(0); s < aut1[0].size(); ++s)
	      pit1[q1][s] /= pitrowsum;
	}

      Gtmp = new PFSA (G);
      Gtmp->set_pit(pit1);
    }
  else 
    Gtmp = new PFSA ();
  return *Gtmp;
}
//--------------------------------------------------------------------------
matrix_dbl llk_distance(vector<symbol_list_>& S)
{
  vector<PFSA> G;
  return llk_distance(S,G);
}
//--------------------------------------------------------------------------
matrix_dbl llk_distance(vector<symbol_list_>& S,
			vector<PFSA>& G,
			unsigned int alphabet)
{
  //get alphabet
  //unsigned int alphabet=0;
  if(alphabet==0)
    for(unsigned int i=0;i<S.size();++i)
      {
	Symbolic_string_ s(S[i]);
	symbol alph(s.get_alphabet());
	if(s.get_alphabet()>alphabet)
	  alphabet=s.get_alphabet();
      }
  
  if(G.empty())
    {
      
      G.push_back(SCC_UTIL__::generate_mc(alphabet,
					  alphabet,
					  "M"));
      G.push_back(SCC_UTIL__::generate_mc(alphabet,
					  alphabet*alphabet,
					  "M"));
      G.push_back(SCC_UTIL__::generate_mc(alphabet,
					  alphabet,
					  "S"));
      G.push_back(SCC_UTIL__::generate_mc(alphabet,
					  alphabet*2,
					  "T"));
    }

  vector<vector<double> > F(G.size(),vector<double> (S.size(),0.0));

  //#pragma omp parallel for 
  for(unsigned int mc=0;mc<G.size();++mc)
    F[mc]= G[mc].log_likelihood(S);

  /*  for(unsigned int mc=0;mc<G.size();++mc)
    {
      for(unsigned int i=0;i<s.size();++i)
	cout << F[mc][i]  << " ";
      cout << endl;
    }
  */
  matrix_dbl M;
  
  for(unsigned int i=0;i<S.size();++i)
    {
      for(unsigned int j=0;j<i;++j)
	{
	  double s=0.0;
	  if(G.empty())
	    M[i][j]=0.0;
	  for(unsigned int k=0;k<G.size();++k)
	    s+=fabs(F[k][i]-F[k][j]);
	  M[i][j]=s/G.size();
	  M[j][i]=M[i][j];
	}
      M[i][i]=0.0;
    }
  
  return M;
};

//------------------------------------
bool SCC_UTIL__::merge_states(PFSA& G,
			      double eps)
{
  pitilde pit=G.get_pit();
  connx aut=G.get_aut();
  unsigned int alphabet=aut[0].size();
  unsigned int numstates=aut.size();

  
  bool BRK=false;
  state q1=0,q2=0;
  for(q1=0;q1<(int)aut.size();++q1)
    {
      for(q2=q1+1;q2<(int)aut.size();++q2)
	if(vec_distance(pit[q1],pit[q2])< eps)
	  {
	    BRK=true;
	    break;
	  }
      if(BRK)
	break;
    }
  
  if(q1==q2)
    return false;
  if((q1>=(int)numstates) || (q2>=(int)numstates))
    return false;
  if (vec_distance(pit[q1],pit[q2])>= eps)
    return false;

  map <state,state > M;
  M[q1]=q1;
  M[q2]=q1;

  //cout << "***** " << q1 << " " << q2 << endl;

  connx aut1;
  for(state q=0;q<(int)aut.size();++q)
    for(symbol s(0);s<alphabet;++s)
      if(M.find(aut[q][s])!=M.end())
	aut1[q][s]=M.find(aut[q][s])->second;
      else
	aut1[q][s]=aut[q][s];

  PFSA G1(pit,aut1);
  G=~G1;

  return true;
};
//-------------------------------------------------


//--------------------------------------------------------------------------
//------------------------------------
PFSA& PFSA::operator ~()
{
  PFSA *Gtmp;
  if (!aut.empty() && !pit.empty())
    {
      typedef adjacency_list<vecS, 
			     vecS,
			     bidirectionalS, 
			     no_property,
			     property<edge_weight_t, float> > Graph;
      typedef std::pair<int,int> Edge;

      int states = aut.size();
      unsigned int num_neg_entries = 0;
      alphabet=aut[0].size();
   
      for (state i = 0;i < states; i++)
	for (symbol j(0);j < alphabet;j++)
	  if (aut[i][j] == -1)
	    num_neg_entries += 1;

      const int num_vertices = states;
      unsigned int num_edges = num_vertices*alphabet - num_neg_entries;
      int edge_weight[num_edges];
      Edge edge_array[num_edges];
      unsigned int pos = 0;

      for (int i = 0; i < states; i++)
	for (symbol j(0); j <  alphabet; j++)
	  if (aut[i][j] != -1)
	    {
	      edge_array[pos] = Edge(i,aut[i][j]);
	      edge_weight[pos++] = j;
	    }
      //create boost graph
      Graph g(edge_array, edge_array + num_edges, edge_weight, num_vertices);
  
      typedef graph_traits<Graph>::vertex_descriptor Vertex;
      vector<int> component(num_vertices), discover_time(num_vertices);
      vector<default_color_type> color(num_vertices);
      vector<Vertex> root(num_vertices);

      /*      int num_components = strong_components(g, &component[0], 
					     root_map(&root[0]).
					     color_map(&color[0]).
					     discover_time_map(&discover_time[0]));

					     // boost 1.48 old code
      */

      int num_components = 
	strong_components(g, 
			  make_iterator_property_map(component.begin(), 
						     get(vertex_index, g)), 
			  root_map(make_iterator_property_map(root.begin(), 
							      get(vertex_index, g))).
			  color_map(make_iterator_property_map(color.begin(), 
							       get(vertex_index, g))).
			  discover_time_map(make_iterator_property_map(discover_time.begin(),
								       get(vertex_index, g))));
      if (num_components == 0)//no strong component present
	Gtmp = new PFSA ();
      else
	{
 	  map <int, vector <state> > componentMap;
	  for (int i=0; i < states;++i)
	    componentMap[component[i]].push_back(i);
 
	  unsigned int sz=0;
	  int cindex=0;
 
	  for (map <int, vector <state> >::iterator itr=componentMap.begin();
	       itr!=componentMap.end();
	       ++itr)
	    if (itr->second.size() > sz)
	      {
		sz = itr->second.size();
		cindex=itr->first;
	      }
    
	  vector <state> Caut(componentMap[cindex]);
    
	  connx aut1;
	  pitilde pit1, Xpit1;
    
	  map <state,state> stateNamemap;
	  for (unsigned int i=0; i< Caut.size();++i)
	    stateNamemap[Caut[i]]=i;
    
	  for (unsigned int i=0; i< Caut.size();++i)
	    for (symbol s(0);s < alphabet;++s)
	      {  
		aut1[i][s] = stateNamemap[aut[Caut[i]][s]];
		pit1[i].push_back(pit[Caut[i]][s]);
	      }
    
	  Gtmp = new PFSA (pit1,aut1);   

	  if (!Xpit.empty())
	    for (unsigned int i=0; i< Caut.size();++i)
	      Xpit1[i] = Xpit[Caut[i]];
			
	  Gtmp->set_Xpit(Xpit1);			
	}
    }
  else		   
    Gtmp = new PFSA ();

  return *Gtmp;
}
//--------------------------------------------------------------------------

PFSA::PFSA(const PFSA& G)
{
  //lock_guard<mutex> guard(mtx_);
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());

  pit=G.pit;
  Xpit=G.Xpit;
  Gamma=G.Gamma;

  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();
  /*
    for ( map < state,  vector < double > >::iterator itr_tmp=pit.begin();
    itr_tmp!=pit.end();++itr_tmp)
    {
    double sum=0.0;
    for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
    itr_row!=(itr_tmp->second).end();++itr_row)
    {
    if (*itr_row > 0.0)
    *itr_row=1/(*itr_row);
    sum+=*itr_row;
    }
    if (sum>0)
    for(vector < double > ::iterator itr_row=(itr_tmp->second).begin();
    itr_row!=(itr_tmp->second).end();++itr_row)
    (*itr_row)=(*itr_row)/sum;
    }  
  */
  aut=G.aut; // same structure


  alphabet = aut[0].size();
  NUM_STATE=aut.size();


  curr_state=G.curr_state;

  size_t len=G.data.size();
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);

}
//--------------------------------------------------------------------------

PFSA::PFSA(string filename)
{
  //lock_guard<mutex> guard(mtx_);
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());

  ifstream DATA(filename.c_str());
  string line;
  stringstream ss;
  unsigned int pitcount=0;
  unsigned int autcount=0;

  pitilde pit_init;
  connx aut_init;
  state state_init;
  bool state_init_flag=false;
  int len_init;
  bool len_init_flag=false;

  while (getline (DATA, line))
    {
      string ch;
      vector < double > prow;
      map_sym_state tmp;
      ss.clear();
      ss.str ("");
      ss << line;
      ss >> ch;
      if (ch=="$")
	{
	  while (ss.good())
	    {
	      double tmp1=-1;
	      ss >> tmp1;
	      if (tmp1 != -1)
		prow.push_back(tmp1);
	    }
	  pit_init[pitcount++]=prow;
	}
      else
	{
	  if (ch=="@")
	    {
	      symbol count;
	      count=0;
	      while (ss.good())
		{
		  state tmp2;
		  ss >> tmp2;
		  tmp[count]=tmp2;
		  count++;
		}
	      aut_init[autcount++]=tmp;
	    }
	  else
	    {
	      if ( ch == "s"  )
		{
		  if (ss.good())
		    {
		      ss >> state_init;
		      state_init_flag = true;
		    }
		}
	      else
		{
		  if ( ch == "len" )
		    {
		      if (ss.good())
			{
			  ss >> len_init;
			  len_init_flag = true;
			}
		    }
		}
	    }
	}
    }

  if (autcount == pitcount)
    {
      pit = pit_init;
      aut = aut_init;

      alphabet = aut[0].size();
      NUM_STATE=aut.size();


      if (state_init_flag)
	curr_state=state_init;
      else
	{
	  srand ( time(NULL) );
	  curr_state = rand()%(pit.size()-1);
	}
    }
  size_t len = LEN_DEFAULT;
  if (len_init_flag)
    len = (size_t) len_init;
  
  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();
  
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;
	
      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);
  
}

//--------------------------------------------------------------------------

PFSA::PFSA(pitilde pit_init, connx aut_init, state state_init, size_t len)
{
  //lock_guard<mutex> guard(mtx_);
  
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());


  pit=pit_init;
  aut=aut_init;

  alphabet = aut[0].size();
  NUM_STATE=aut.size();


  curr_state=state_init;
  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();

  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);
};
//--------------------------------------------------------------------------
PFSA::PFSA(pitilde pit_init, connx aut_init, state state_init)
{
  //lock_guard<mutex> guard(mtx_);

  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());


  pit=pit_init;
  aut=aut_init;

  alphabet = aut[0].size();
  NUM_STATE=aut.size();


  curr_state=state_init;
  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();

  /** \b Using \b Default data length */
  size_t len = LEN_DEFAULT;
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);
};

//--------------------------------------------------------------------------

PFSA::PFSA(pitilde pit_init, connx aut_init)
{
  //lock_guard<mutex> guard(mtx_);

  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());


  pit=pit_init;
  aut=aut_init;
  alphabet = aut[0].size();
  NUM_STATE=aut.size();

  /** \b Random initial state */
  srand ( time(NULL) );
  if (pit.size() > 1)
    curr_state=rand()%(pit.size()-1);
  else
    curr_state=0;
  
  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();

  /** \b Using \b Default data length */
  size_t len = LEN_DEFAULT;
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);
};
//--------------------------------------------------------------------------
/*void PFSA::mc_print()
{
  //lock_guard<mutex> guard(mtx_);

  if ((!aut.empty()) && (!pit.empty()))
    {
      cout << "PITILDE: size(" << pit.size() << ")" <<  endl;
      for (pitilde::iterator itr=pit.begin();
	   itr!=pit.end();++itr)
	{
	  for (unsigned int j=0; j < itr->second.size(); ++j)
	    cout << (itr->second)[j] << " " ;
	  cout << endl;
	}

      cout << "CONNX: size(" << aut.size() << ")" << endl;
      for (connx::iterator itr=aut.begin();
	   itr!=aut.end();++itr)
	{
	  for (map_sym_state::iterator itr1=itr->second.begin();
	       itr1!=itr->second.end();++itr1)
	    cout << itr1->second << " " ;
	  cout << endl;
	}
      if (!Xpit.empty())
	{
	  cout << "XPIT: Xsize(" << Xpit[0].size() << ")" << endl;
	  for (pitilde::iterator itr=Xpit.begin();
	       itr!=Xpit.end();++itr)
	    {
	      for (unsigned int j=0; j < itr->second.size(); ++j)
		cout << (itr->second)[j] << " " ;
	      cout << endl;
	    }
	}
    }  
}
*/
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void PFSA::mc_print(ostream & out)
{
  //lock_guard<mutex> guard(mtx_);

  if ((!aut.empty()) && (!pit.empty()))
    {
      out << "%PITILDE: size(" << pit.size() << ")" <<  endl;
      out << "#PITILDE " <<  endl;
      for (pitilde::iterator itr=pit.begin();
	   itr!=pit.end();++itr)
	{
	  for (unsigned int j=0; j < itr->second.size(); ++j)
	    out << (itr->second)[j] << " " ;
	  out << endl;
	}

      out << "%CONNX: size(" << aut.size() << ")" << endl;
      out << "#CONNX " << endl;
      for (connx::iterator itr=aut.begin();
	   itr!=aut.end();++itr)
	{
	  for (map_sym_state::iterator itr1=itr->second.begin();
	       itr1!=itr->second.end();++itr1)
	    out << itr1->second << " " ;
	  out << endl;
	}
      if (!Xpit.empty())
	{
	  out << "%XPIT: Xsize(" << Xpit[0].size() << ")" << endl;
	  out << "#XPIT " << endl;
	  for (pitilde::iterator itr=Xpit.begin();
	       itr!=Xpit.end();++itr)
	    {
	      for (unsigned int j=0; j < itr->second.size(); ++j)
		out << (itr->second)[j] << " " ;
	      out << endl;
	    }
	}
    }  
}
//-----------------------------------------------------------
PFSA& PFSA::operator= (const PFSA & rhs)
{
  //lock_guard<mutex> guard(mtx_);

  // check for self-assignment by comparing the address of the
  // implicit object and the parameter
  if (this == &rhs)
    return *this;
 
  // do the copy   
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());

  pit=rhs.pit;
  Xpit=rhs.Xpit;
  Gamma=rhs.Gamma;
  
  map < state,  vector < double > >::iterator itr=pit.begin();
  size_t alph = (itr->second).size();
  aut=rhs.aut; // same structure

  alphabet = aut[0].size();
  NUM_STATE=aut.size();

  curr_state=rhs.curr_state;

  size_t len=rhs.data.size();
  for(size_t i = 0; i < len; i++)
    {
      //   vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + pit[curr_state][sym];
	  if (key <= cumprob)
	    break;
	}
      if(aut[curr_state][sym]!=-1)
	{
	  data.push_back(sym);
	  curr_state=aut[curr_state][sym];
	}
      else
	break;
    }
  gsl_rng_free (r);

  // return the existing object
  return *this;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void PFSA::drawGraphX(string dotfile, string dotcfg, string Title)
{
  //lock_guard<mutex> guard(mtx_);
  string graphsize = "40,25", 
    rankdir = "TD", 
    nodeshape = "circle",
    DPI="600",
    bgcolor="transparent",
    nodestyle = "filled", 
    nodecolor = "black", 
    nodefontsize="16",
    edgeweight = "16",
    edgefontcolor = "midnightblue", 
    edgefontweight = "bold",
    nodefontweight = "bold",
    nodefontcolor="black",
    edgestyle = "solid", 
    edgecolor = "darkslategray4", 
    edgefontname = "Helvetica",
    nodefontname = "Helvetica",
    nodefillcolor = "papayawhip", 
    graphname = "PFSA",
    labelfontcolor="black",
    dotprog="dot",
    edgepenwidth="1",
    arrowsize="1",
    taglinkcolor="black",
    tagbordercolor="black",
    tagfillcolor="whitesmoke",
    Type="png";
  
  CONFIG cfg(dotcfg);
  cfg.set(graphsize,"GRAPH_SIZE");
  cfg.set(rankdir,"RANKDIR");
  cfg.set(DPI,"DPI");
  cfg.set(bgcolor,"BGCOLOR");
  cfg.set(nodestyle,"NODESTYLE");
  cfg.set(nodecolor,"NODECOLOR");
  cfg.set(nodefontsize,"NODEFONTSIZE");
  cfg.set(nodefontweight,"NODE_FONTWEIGHT");
  cfg.set(edgeweight,"EDGE_WEIGHT");
  cfg.set(edgefontcolor,"EDGE_FONTCOLOR");
  cfg.set(nodefontweight,"EDEGE_FONTWEIGHT");
  cfg.set(nodefontcolor,"NODE_FONTCOLOR");
  cfg.set(edgestyle,"EDGE_STYLE");
  cfg.set(edgefontname,"EDGE_FONTNAME");
  cfg.set(nodefontname,"NODE_FONTNAME");
  cfg.set(labelfontcolor,"LABEL_COLOR");
  cfg.set(edgecolor,"EDGE_COLOR");
  cfg.set(edgepenwidth,"EDGE_PENWIDTH");
  cfg.set(nodefillcolor,"NODE_FILLCOLOR");
  cfg.set(graphname,"GRAPH_NAME");
  cfg.set(arrowsize,"ARROW_SIZE");
  cfg.set(taglinkcolor,"TAGLINK_COLOR");
  cfg.set(tagbordercolor,"TAGBORDER_COLOR");
  cfg.set(tagfillcolor,"TAGFILL_COLOR");
  cfg.set(Type,"GRAPH_TYPE");
  cfg.set(dotprog,"DOT_PROG");

  if (aut.size() > 0)
    {
      NUM_STATE=aut.size();
      alphabet=aut[0].size();

      Digraph g(NUM_STATE);
      symbol s;

      vector < string >  name_string_arr;
      string  name[2*NUM_STATE];
      for (unsigned int i = 0; i < NUM_STATE; i++)
	{
	  char buff[2048];
	  sprintf(buff,"<q<SUB>%d</SUB>>",i);
	  name[i] =  buff;
	} 
      for (unsigned int i = 0; i < NUM_STATE; i++)
	{
	  char buff[2048];
	  string strtmp ="\"[";
	  for (s=0; s < pit[i].size(); s++)
	    {
	      sprintf(buff,"%1.2f",pit[i][s]);
	      strtmp += buff;
	      if (s+1==pit[i].size())
		{
		  strtmp += "]\", shape=box,fontcolor=black, fontname=Helvetica, fontsize=14, penwidth=2, fillcolor=";
		  strtmp+=tagfillcolor+",color="+tagbordercolor;
		}
	      else
		strtmp += " ";
	    }
	  name[i+NUM_STATE]=strtmp;
	} 
      
      property_map<Digraph, edge_name_t>::type namemap = get(edge_name, g);
      int edgecount=0;
      for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
	for (s=0; s < alphabet; s++)
	  if (aut[num_state][s] != -1)
	    {
	      char buff1[2048];
	      sprintf(buff1,"%d",(int)s);
	      string strtmp ="label=";
	      strtmp += buff1;
	      name_string_arr.push_back(strtmp);	  
	      graph_traits<Digraph>::edge_descriptor e; 
	      bool inserted;
	      tie(e, inserted) = add_edge( num_state,  aut[num_state][s], g);
	      namemap[e] = name_string_arr[edgecount++];
	    }

      for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
	{
	  string strtmp ="arrowhead=none, penwidth=3,color=";
	  strtmp += taglinkcolor;
	  name_string_arr.push_back(strtmp);
	  graph_traits<Digraph>::edge_descriptor e; 
	  bool inserted;
	  tie(e, inserted) = add_edge( num_state,  NUM_STATE+num_state, g);
	  namemap[e] = name_string_arr[edgecount++];
	}
      
      boost::property_map<Digraph, edge_name_t>::type  edge_name_arr = get(edge_name, g);
      std::map<std::string,std::string> graph_attr, vertex_attr, edge_attr;


      string Str1=dotprog + " -T";
      string DCOM;

      if (Title != "")
	{
	  graph_attr["labelloc"] = "b";
	  graph_attr["label"] = Title;
	  graph_attr["fontcolor"] = labelfontcolor;
	}
      graph_attr["bgcolor"] = bgcolor;
      graph_attr["size"] = graphsize;
      graph_attr["rankdir"] = rankdir;
      graph_attr["dpi"] = DPI;
      //	  graph_attr["ratio"] = "fill";
      vertex_attr["shape"] = nodeshape;
      vertex_attr["style"] = nodestyle;
      vertex_attr["color"] = nodecolor;//"indianred";
      vertex_attr["fillcolor"] = nodefillcolor;//"indianred";
      vertex_attr["fontcolor"] = nodefontcolor;//"indianred";
      vertex_attr["fontweight"] = nodefontweight;//"indianred";
      vertex_attr["fontsize"] = nodefontsize;//"indianred";
      vertex_attr["fontname"] = nodefontname;//"indianred";
      edge_attr["fontcolor"] = edgefontcolor;
      edge_attr["fontname"] = edgefontname;
      edge_attr["fontweight"] = edgefontweight;
      edge_attr["weight"] = edgeweight;
      edge_attr["style"] = edgestyle;
      edge_attr["color"] = edgecolor;
      edge_attr["penwidth"] = edgepenwidth;
      edge_attr["arrowsize"] = arrowsize;

      ofstream outfile((dotfile+".dot").c_str());

      write_graphviz(outfile, g,	
		     make_label_writer_html(name), 
		     make_label_writer_noquotes(edge_name_arr),
		     make_graph_attributes_writer(graph_attr, 
						  vertex_attr, 
						  edge_attr ));
      outfile.close();
      if (Type != "NO_DOT")
	{
	  DCOM=Str1+Type+ " " + dotfile +".dot -o"+ " " + dotfile+"."+Type; 
	  std::system(DCOM.c_str());
	}
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void PFSA::drawGraph(string dotfile, string dotcfg, string Title)
{
  //lock_guard<mutex> guard(mtx_);
  double MIN_EDGE_PROB = 0.01;
  if (aut.size() > 0)
    {
      Digraph g(NUM_STATE);

      vector < string >  name_string_arr;
      string  name[NUM_STATE];
      for (unsigned int i = 0; i < NUM_STATE; i++)
	{
	  char buff[2048];
	  sprintf(buff,"<q<SUB>%d</SUB>>",i);//<Regular<SUB>subscript</SUB>>
	  name[i] =  buff;
	} 

      property_map<Digraph, edge_name_t>::type namemap = get(edge_name, g);
      int edgecount=0;
      symbol s;
      for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
	for (s=0; s < alphabet; s++)
	  if ((aut[num_state][s] != -1) && (pit[num_state][s] >= MIN_EDGE_PROB ))
	    {
	      char buff1[2048],buff2[1024];
	      sprintf(buff1,"%d",(int)s);
	      sprintf(buff2,"(%1.2f)",pit[num_state][s]);
	      string strtmp ="";
	      strtmp += buff1;
	      strtmp += buff2;
	      name_string_arr.push_back(strtmp);	  
	      graph_traits<Digraph>::edge_descriptor e; 
	      bool inserted;
	      tie(e, inserted) = add_edge( num_state,  aut[num_state][s], g);
	      namemap[e] = name_string_arr[edgecount++];
	    }

      boost::property_map<Digraph, edge_name_t>::type  edge_name_arr = get(edge_name, g);
      std::map<std::string,std::string> graph_attr, vertex_attr, edge_attr;


      string graphsize = "40,25", 
	rankdir = "TD", 
	nodeshape = "circle",
	DPI="600",
	nodestyle = "filled", 
	nodecolor = "powderblue", 
	nodefontsize="16",
	bgcolor="transparent",
	edgeweight = "20",
	edgefontcolor = "midnightblue", 
	edgefontweight = "bold",
	nodefontweight = "bold",
	nodefontcolor="black",
	edgestyle = "solid", 
	edgecolor = "slateblue1", 
	edgefontname = "courier",
	nodefontname = "courier",
	nodefillcolor = "mintcream", 
	graphname = "PFSA",
	dotprog="dot",
	edgepenwidth="1",
	arrowsize="1",
	labelfontcolor="black",
	Type="png";

      CONFIG cfg(dotcfg);
      cfg.set(graphsize,"GRAPH_SIZE");
      cfg.set(rankdir,"RANKDIR");
      cfg.set(DPI,"DPI");
      cfg.set(nodestyle,"NODESTYLE");
      cfg.set(nodecolor,"NODECOLOR");
      cfg.set(bgcolor,"BGCOLOR");
      cfg.set(nodefontsize,"NODEFONTSIZE");
      cfg.set(nodefontweight,"NODE_FONTWEIGHT");
      cfg.set(edgeweight,"EDGE_WEIGHT");
      cfg.set(edgefontcolor,"EDGE_FONTCOLOR");
      cfg.set(nodefontweight,"EDEGE_FONTWEIGHT");
      cfg.set(nodefontcolor,"NODE_FONTCOLOR");
      cfg.set(edgestyle,"EDGE_STYLE");
      cfg.set(edgefontname,"EDGE_FONTNAME");
      cfg.set(nodefontname,"NODE_FONTNAME");
      cfg.set(edgecolor,"EDGE_COLOR");
      cfg.set(labelfontcolor,"LABEL_COLOR");
      cfg.set(edgepenwidth,"EDGE_PENWIDTH");
      cfg.set(nodefillcolor,"NODE_FILLCOLOR");
      cfg.set(graphname,"GRAPH_NAME");
      cfg.set(arrowsize,"ARROW_SIZE");
      cfg.set(Type,"GRAPH_TYPE");
      cfg.set(dotprog,"DOT_PROG");

      string Str1=dotprog + " -T";
      string DCOM;

      if (Title != "")
	{
	  graph_attr["labelloc"] = "b";
	  graph_attr["label"] = Title;
	  graph_attr["fontcolor"] = labelfontcolor;
	}
      graph_attr["bgcolor"] = bgcolor;
      graph_attr["size"] = graphsize;
      graph_attr["rankdir"] = rankdir;
      graph_attr["dpi"] = DPI;
      //	  graph_attr["ratio"] = "fill";
      vertex_attr["shape"] = nodeshape;
      vertex_attr["style"] = nodestyle;
      vertex_attr["color"] = nodecolor;//"indianred";
      vertex_attr["fillcolor"] = nodefillcolor;//"indianred";
      vertex_attr["fontcolor"] = nodefontcolor;//"indianred";
      vertex_attr["fontweight"] = nodefontweight;//"indianred";
      vertex_attr["fontsize"] = nodefontsize;//"indianred";
      vertex_attr["fontname"] = nodefontname;//"indianred";
      edge_attr["fontcolor"] = edgefontcolor;
      edge_attr["fontname"] = edgefontname;
      edge_attr["fontweight"] = edgefontweight;
      edge_attr["fontsize"] = nodefontsize;
      edge_attr["weight"] = edgeweight;
      edge_attr["style"] = edgestyle;
      edge_attr["color"] = edgecolor;
      edge_attr["penwidth"] = edgepenwidth;
      edge_attr["arrowsize"] = arrowsize;


      ofstream outfile((dotfile+".dot").c_str());

      write_graphviz(outfile, g,	
		     make_label_writer_html(name), make_label_writer(edge_name_arr),
		     make_graph_attributes_writer(graph_attr, vertex_attr, 
						  edge_attr ));
      outfile.close();
      if (Type != "NO_DOT")
	{
	  DCOM=Str1+Type+ " " + dotfile +".dot -o"+ " " + dotfile+"."+Type; 
	  std::system(DCOM.c_str());
	}
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void PFSA::drawGraph(string dotfile)
{
  //lock_guard<mutex> guard(mtx_);
  Digraph g(NUM_STATE);

  vector < string >  name_string_arr;
  string  name[NUM_STATE];
  for (unsigned int i = 0; i < NUM_STATE; i++)
    {
      char buff[2048];
      sprintf(buff,"<q<SUB>%d</SUB>>",i);//<Regular<SUB>subscript</SUB>>
      name[i] =  buff;
    } 

  property_map<Digraph, edge_name_t>::type namemap = get(edge_name, g);
  int edgecount=0;
  symbol s;
  for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
    for (s=0; s < alphabet; s++)
      if (aut[num_state][s] != -1)
	{
	  char buff1[2048],buff2[1024];
	  sprintf(buff1,"%d",(int)s);
	  sprintf(buff2,"(%1.2f)",pit[num_state][s]);
	  string strtmp ="";
	  strtmp += buff1;
	  strtmp += buff2;
	  name_string_arr.push_back(strtmp);	  
	  graph_traits<Digraph>::edge_descriptor e; 
	  bool inserted;
	  tie(e, inserted) = add_edge( num_state,  aut[num_state][s], g);
	  namemap[e] = name_string_arr[edgecount++];
	}

  boost::property_map<Digraph, edge_name_t>::type  edge_name_arr = get(edge_name, g);
  std::map<std::string,std::string> graph_attr, vertex_attr, edge_attr;


  string graphsize = "40,25", 
    rankdir = "TD", 
    nodeshape = "circle",
    DPI="600",
    nodestyle = "filled", 
    nodecolor = "powderblue", 
    nodefontsize="16",
    edgeweight = "20",
    edgefontcolor = "midnightblue", 
    edgefontweight = "bold",
    nodefontweight = "bold",
    nodefontcolor="black",
    edgestyle = "solid", 
    edgecolor = "slateblue1", 
    edgefontname = "courier",
    nodefontname = "courier",
    nodefillcolor = "mintcream", 
    graphname = "PFSA",
    dotprog="dot",
    edgepenwidth="1",
    arrowsize="1",
    Type="png";
  
  CONFIG cfg(dotfile);
  cfg.set(graphsize,"GRAPH_SIZE");
  cfg.set(rankdir,"RANKDIR");
  cfg.set(DPI,"DPI");
  cfg.set(nodestyle,"NODESTYLE");
  cfg.set(nodecolor,"NODECOLOR");
  cfg.set(nodefontsize,"NODEFONTSIZE");
  cfg.set(nodefontweight,"NODE_FONTWEIGHT");
  cfg.set(edgeweight,"EDGE_WEIGHT");
  cfg.set(edgefontcolor,"EDGE_FONTCOLOR");
  cfg.set(nodefontweight,"EDEGE_FONTWEIGHT");
  cfg.set(nodefontcolor,"NODE_FONTCOLOR");
  cfg.set(edgestyle,"EDGE_STYLE");
  cfg.set(edgefontname,"EDGE_FONTNAME");
  cfg.set(nodefontname,"NODE_FONTNAME");
  cfg.set(edgecolor,"EDGE_COLOR");
  cfg.set(edgepenwidth,"EDGE_PENWIDTH");
  cfg.set(nodefillcolor,"NODE_FILLCOLOR");
  cfg.set(graphname,"GRAPH_NAME");
  cfg.set(arrowsize,"ARROW_SIZE");
  cfg.set(Type,"GRAPH_TYPE");
  cfg.set(dotprog,"DOT_PROG");

  string Str1=dotprog + " -T";
  string DCOM;

  graph_attr["bgcolor"] = "transparent";
  graph_attr["size"] = graphsize;
  graph_attr["rankdir"] = rankdir;
  graph_attr["dpi"] = DPI;
  //	  graph_attr["ratio"] = "fill";
  vertex_attr["shape"] = nodeshape;
  vertex_attr["style"] = nodestyle;
  vertex_attr["color"] = nodecolor;//"indianred";
  vertex_attr["fillcolor"] = nodefillcolor;//"indianred";
  vertex_attr["fontcolor"] = nodefontcolor;//"indianred";
  vertex_attr["fontweight"] = nodefontweight;//"indianred";
  vertex_attr["fontsize"] = nodefontsize;//"indianred";
  vertex_attr["fontname"] = nodefontname;//"indianred";
  edge_attr["fontcolor"] = edgefontcolor;
  edge_attr["fontname"] = edgefontname;
  edge_attr["fontweight"] = edgefontweight;
  edge_attr["weight"] = edgeweight;
  edge_attr["style"] = edgestyle;
  edge_attr["color"] = edgecolor;
  edge_attr["penwidth"] = edgepenwidth;
  edge_attr["arrowsize"] = arrowsize;


  
  ofstream outfile("Inferred_mc.dot");
  write_graphviz(outfile, g,	
		 make_label_writer_html(name), make_label_writer(edge_name_arr),
		 make_graph_attributes_writer(graph_attr, vertex_attr, 
					      edge_attr ));
  outfile.close();
  if (Type != "NO_DOT")
    {
      DCOM=Str1+Type+ " " + "Inferred_mc.dot -o"+ " " + "Inferred_mc"+"."+Type; 
      std::system(DCOM.c_str());
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void PFSA::drawGraph()
{
  //lock_guard<mutex> guard(mtx_);
  Digraph g(NUM_STATE);

  vector < string >  name_string_arr;
  string  name[NUM_STATE];
  for (unsigned int i = 0; i < NUM_STATE; i++)
    {
      char buff[2048];
      sprintf(buff,"<q<SUB>%d</SUB>>",i);//<Regular<SUB>subscript</SUB>>
      name[i] =  buff;
    } 

  property_map<Digraph, edge_name_t>::type namemap = get(edge_name, g);
  int edgecount=0;
  symbol s;
  for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
    for (s=0; s < alphabet; s++)
      if (aut[num_state][s] != -1)
	{
	  char buff1[2048],buff2[1024];
	  sprintf(buff1,"%d",(int)s);
	  sprintf(buff2,"(%1.2f)",pit[num_state][s]);
	  string strtmp ="";
	  strtmp += buff1;
	  strtmp += buff2;
	  name_string_arr.push_back(strtmp);	  
	  graph_traits<Digraph>::edge_descriptor e; 
	  bool inserted;
	  tie(e, inserted) = add_edge( num_state,  aut[num_state][s], g);
	  namemap[e] = name_string_arr[edgecount++];
	}

  boost::property_map<Digraph, edge_name_t>::type  edge_name_arr = get(edge_name, g);
  std::map<std::string,std::string> graph_attr, vertex_attr, edge_attr;


  string graphsize = "40,25", 
    rankdir = "TD", 
    nodeshape = "circle",
    DPI="600",
    nodestyle = "filled", 
    nodecolor = "powderblue", 
    nodefontsize="16",
    edgeweight = "20",
    edgefontcolor = "midnightblue", 
    edgefontweight = "bold",
    nodefontweight = "bold",
    nodefontcolor="black",
    edgestyle = "solid", 
    edgecolor = "slateblue1", 
    edgefontname = "courier",
    nodefontname = "courier",
    nodefillcolor = "mintcream", 
    graphname = "PFSA",
    dotprog="dot",
    edgepenwidth="1",
    arrowsize="1",
    Type="png";
  
  CONFIG cfg("dot.cfg");
  cfg.set(graphsize,"GRAPH_SIZE");
  cfg.set(rankdir,"RANKDIR");
  cfg.set(DPI,"DPI");
  cfg.set(nodestyle,"NODESTYLE");
  cfg.set(nodecolor,"NODECOLOR");
  cfg.set(nodefontsize,"NODEFONTSIZE");
  cfg.set(nodefontweight,"NODE_FONTWEIGHT");
  cfg.set(edgeweight,"EDGE_WEIGHT");
  cfg.set(edgefontcolor,"EDGE_FONTCOLOR");
  cfg.set(nodefontweight,"EDEGE_FONTWEIGHT");
  cfg.set(nodefontcolor,"NODE_FONTCOLOR");
  cfg.set(edgestyle,"EDGE_STYLE");
  cfg.set(edgefontname,"EDGE_FONTNAME");
  cfg.set(nodefontname,"NODE_FONTNAME");
  cfg.set(edgecolor,"EDGE_COLOR");
  cfg.set(edgepenwidth,"EDGE_PENWIDTH");
  cfg.set(nodefillcolor,"NODE_FILLCOLOR");
  cfg.set(graphname,"GRAPH_NAME");
  cfg.set(arrowsize,"ARROW_SIZE");
  cfg.set(Type,"GRAPH_TYPE");
  cfg.set(dotprog,"DOT_PROG");

  string Str1=dotprog + " -T";
  string DCOM;

  graph_attr["bgcolor"] = "transparent";
  graph_attr["size"] = graphsize;
  graph_attr["rankdir"] = rankdir;
  graph_attr["dpi"] = DPI;
  //	  graph_attr["ratio"] = "fill";
  vertex_attr["shape"] = nodeshape;
  vertex_attr["style"] = nodestyle;
  vertex_attr["color"] = nodecolor;//"indianred";
  vertex_attr["fillcolor"] = nodefillcolor;//"indianred";
  vertex_attr["fontcolor"] = nodefontcolor;//"indianred";
  vertex_attr["fontweight"] = nodefontweight;//"indianred";
  vertex_attr["fontsize"] = nodefontsize;//"indianred";
  vertex_attr["fontname"] = nodefontname;//"indianred";
  edge_attr["fontcolor"] = edgefontcolor;
  edge_attr["fontname"] = edgefontname;
  edge_attr["fontweight"] = edgefontweight;
  edge_attr["weight"] = edgeweight;
  edge_attr["style"] = edgestyle;
  edge_attr["color"] = edgecolor;
  edge_attr["penwidth"] = edgepenwidth;
  edge_attr["arrowsize"] = arrowsize;

  string dotfile="Inferred_mc";

  ofstream outfile((dotfile+".dot").c_str());
  write_graphviz(outfile, g,	
		 make_label_writer_html(name), make_label_writer(edge_name_arr),
		 make_graph_attributes_writer(graph_attr, vertex_attr, 
					      edge_attr ));
  outfile.close();
  if (Type != "NO_DOT")
    {
      DCOM=Str1+Type+ " " + dotfile +".dot -o"+ " " + dotfile+"."+Type; 
      std::system(DCOM.c_str());
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
Symbolic_string_  PFSA::gen_data(size_t len1)
{
  //lock_guard<mutex> guard(mtx_);
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());
  /** \b Random initial state */
  srand ( time(NULL) );
  if (aut.size() > 1)
    curr_state=rand()%(aut.size()-1);
  else
    curr_state=0;

  size_t alph = pit[0].size(); 
  /** \b Using  given data length */

  symbol_list_ s(len1);
  for(size_t i = 0; i < len1; i++)
    {
      // vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + pit[curr_state][sym];
	  if (key <= cumprob)
	    break;
	}
      s[i] = sym;
      curr_state=aut[curr_state][sym];
    }
  gsl_rng_free (r);
  return Symbolic_string_(s,alph);
};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
Symbolic_string_  PFSA::gen_data()
{
  //lock_guard<mutex> guard(mtx_);
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());
  /** \b Random initial state */
  srand ( time(NULL) );
  if (aut.size() > 1)
    curr_state=rand()%(aut.size()-1);
  else
    curr_state=0;
  size_t alph = pit[0].size();
  /** \b Using \b Default data length */
  size_t len = LEN_DEFAULT;

  symbol_list_ s(len);
  for(size_t i = 0; i < len; i++)
    {
      vector < double > prow(pit[curr_state]);
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alph-1; sym++)
	{
	  cumprob = cumprob + prow[sym];
	  if (key <= cumprob)
	    break;
	}
      s[i]=sym;
      curr_state=aut[curr_state][sym];
    }
  gsl_rng_free (r);
  return Symbolic_string_(s,alph);
};

//--------------------------------------------------------------------------
void PFSA::gen_PI(void)
{
  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);
  if (!empty())
    {
      PImat = Z_sq_row;
      symbol j;
      state i;
      for (i=0;i< (int) aut.size();i++)
	for (j=0; j <aut[0].size();j++)
	  if (aut[i][j] != -1)
	    PImat[i][aut[i][j]] += pit[i][j];
    }
}
//--------------------------------------------------------------------------
void PFSA::gen_Stationary(void)
{
  //compute approximate stationary state distribution
  unsigned int ITER=10000;
  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);
  Stationary_distribution = Zrow;

  if (!empty())
    {
      if (PImat.empty())
	gen_PI();

      vector <vector <double> > J(Z_sq_row);
      vector <vector <double> > J_new(Z_sq_row);

      for (unsigned int i=0;i<aut.size();i++)
	J[i][i] = 1.0;

      vector <vector <double> > Jsum(J);

      for (unsigned int n=0; n<ITER;++n)
	{
	  for(unsigned int i=0;i<aut.size();i++)
	    for(unsigned int k=0;k<aut.size();k++)
	      {	      
		double S=0.0;
		for(unsigned int j=0;j<aut.size();j++)
		  S+=J[i][j]*PImat[j][k];
		J_new[i][k]=S;
	      }		  
	  J=J_new;

	  for(unsigned int i=0;i<aut.size();i++)
	    for(unsigned int k=0;k<aut.size();k++)
	      Jsum[i][k] += J[i][k];
	}

      for(unsigned int i=0;i<aut.size();i++)
	for(unsigned int k=0;k<aut.size();k++)
	  Jsum[i][k] /= (ITER+0.0);

      for(unsigned int i=0;i<aut.size();i++)
	Stationary_distribution[i] = Jsum[i][i];

      double S=0.0;
      for (unsigned int i=0;i<aut.size();i++)
	S+=Stationary_distribution[i];
      for (unsigned int i=0;i<aut.size();i++)
	Stationary_distribution[i] /= S;
    }   
}
//--------------------------------------------------------------------------
/*
void PFSA::gen_Stationary(int ITER)
{
  //compute approximate stationary state distribution
  double theta=0.0001;
  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);


  if (!empty())
    {
      if (PImat.empty())
	gen_PI();

      vector <vector <double> > J_new(Z_sq_row);
      vector <vector <double> > J(Z_sq_row);
      for (unsigned int i=0;i<aut.size();i++)
	J[i][i] = 1.0;
      vector <vector <double> > Q(Z_sq_row);
      for (unsigned int i=0;i<aut.size();i++)
	Q[i][i] = theta;
      for (unsigned int n=0; n<ITER; n++)
	{
	  for(unsigned int i=0;i<aut.size();i++)
	    for(unsigned int k=0;k<aut.size();k++)
	      {	      
		double S=0.0;
		for(unsigned int j=0;j<aut.size();j++)
		  S+=J[i][j]*PImat[j][k];
		J_new[i][k]=S;
	      }	  
	  for (unsigned int ii=0;ii<aut.size();ii++)
	    for (unsigned int jj=0; jj <aut.size();jj++)
	      J[ii][jj] = (1-theta)*J_new[ii][jj];

	  for (unsigned int ii=0;ii<aut.size();ii++)
	    for (unsigned int jj=0; jj <aut.size();jj++)
	      Q[ii][jj] += theta*J[ii][jj];
	}
      Stationary_distribution = Q[0];

      double S=0.0;
      for (unsigned int i=0;i<aut.size();i++)
	S+=Stationary_distribution[i];
      for (unsigned int i=0;i<aut.size();i++)
	Stationary_distribution[i] /= S;
    }
  else
    Stationary_distribution = Zrow;
}
*/
//--------------------------------------------------------------------------

void PFSA::gen_Stationary(unsigned int ITER)
{
  //compute approximate stationary state distribution
  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);
  Stationary_distribution = Zrow;

  if (!empty())
    {
      if (PImat.empty())
	gen_PI();

      vector <vector <double> > J(Z_sq_row);
      vector <vector <double> > J_new(Z_sq_row);

      for (unsigned int i=0;i<aut.size();i++)
	J[i][i] = 1.0;

      vector <vector <double> > Jsum(J);

      for (unsigned int n=0; n<ITER;++n)
	{
          #pragma omp parallel for schedule(dynamic,1) collapse(2)
	  for(unsigned int i=0;i<aut.size();i++)
	    for(unsigned int k=0;k<aut.size();k++)
	      {	      
		double S=0.0;
		for(unsigned int j=0;j<aut.size();j++)
		  S+=J[i][j]*PImat[j][k];
		J_new[i][k]=S;
	      }		  
	  J=J_new;

	  for(unsigned int i=0;i<aut.size();i++)
	    for(unsigned int k=0;k<aut.size();k++)
	      Jsum[i][k] += J[i][k];
	}

      for(unsigned int i=0;i<aut.size();i++)
	for(unsigned int k=0;k<aut.size();k++)
	  Jsum[i][k] /= (ITER+0.0);

      for(unsigned int i=0;i<aut.size();i++)
	Stationary_distribution[i] = Jsum[i][i];

      double S=0.0;
      for (unsigned int i=0;i<aut.size();i++)
	S+=Stationary_distribution[i];
      for (unsigned int i=0;i<aut.size();i++)
	Stationary_distribution[i] /= S;
    }   
}
//--------------------------------------------------------------------------

void PFSA::gen_Gamma(void)
{
  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);
  symbol sym;
  for (sym=0; sym< aut[0].size(); sym++)
    {
      vector <vector <double> > gtmp(aut.size(),Zrow);
      for (unsigned int i=0;i<aut.size();i++)
	if (aut[i][sym] != -1)
	  gtmp[i][aut[i][sym]] += pit[i][sym];
      Gamma[sym] = gtmp;
    }
}

//--------------------------------------------------------------------------
vector <double>  PFSA::run_graph(symbol_list_ history)
{
  srand ( time(NULL) );
  if (aut.size() > 1)
    curr_state=rand()%(aut.size()-1);
  else
    curr_state=0;

  vector <double> state_visit(aut.size(),0.0);
  unsigned int len=history.size();
  for(size_t i = 0; i < len; i++)
    {
      state_visit[curr_state]++;
      curr_state=aut[curr_state][history[i]];
    }

  for(size_t i = 0; i < aut.size(); i++)
    state_visit[i]/=(len+0.0);

  return state_visit;
}
//--------------------------------------------------------------------------
vector <vector <double> >  PFSA::run(symbol_list_ history, unsigned int future_steps)
{
  if (PImat.empty())
    gen_PI();
  if (Gamma.empty())
    gen_Gamma();
  if (Stationary_distribution.empty())
    gen_Stationary();

  vector <double> Zrow(aut.size(),0.0);
  vector <vector <double> > Z_sq_row(aut.size(),Zrow);

  vector <double> state_vec;
  state_vec = Stationary_distribution;

  vector <vector <double> > G(Z_sq_row);
  for (unsigned int i=0;i<aut.size();i++)
    G[i][i] = 1.0;

  for(size_t sym = 0; sym < history.size(); sym++)
    {
      vector <vector <double> > G_next(Gamma[history[sym]]);
      vector <vector <double> > G_new(aut.size(),Zrow);

      for(unsigned int i=0;i<aut.size();i++)
	for(unsigned int k=0;k<aut.size();k++)
	  {	      
	    double S=0.0;
	    for(unsigned int j=0;j<aut.size();j++)
	      S+=G[i][j]*G_next[j][k];
	    G_new[i][k]=S;
	  }	  
      for (unsigned int ii=0;ii<aut.size();ii++)
	for (unsigned int jj=0; jj <aut.size();jj++)
	  G[ii][jj] = G_new[ii][jj];
    }

  vector <double> state_vec_tmp(state_vec);
  for (unsigned int i=0; i < aut.size(); i++)
    {
      double V=0.000001;
      for (unsigned int j=0; j < aut.size(); j++)
	V+=G[j][i]*state_vec_tmp[j];
      state_vec[i]=V;
    }

  //normalize state_vec
  double S=0.0;
  for (unsigned int i=0;i<aut.size();i++)
    S+=state_vec[i];
  for (unsigned int i=0;i<aut.size();i++)
    state_vec[i] /= S;

  //compute future distrubutions
  vector <vector <double> > F_Dist;

  vector <double> future_dist(aut[0].size());
  vector <double> fdist_curr(aut[0].size(),0.0);
  for (unsigned int i=0; i < aut[0].size(); i++)
    {
      double V=0.0;
      for (unsigned int j=0; j < aut.size(); j++)
	V+=pit[j][i]*state_vec[j];
      fdist_curr[i]=V;
    }
  F_Dist.push_back(fdist_curr);


  for (unsigned int f=1;f <future_steps;f++)
    {
      vector <double> state_vec_tmp(state_vec);
      for (unsigned int i=0; i < aut.size(); i++)
	{
	  double V=0.0;
	  for (unsigned int j=0; j < aut.size(); j++)
	    V+=PImat[j][i]*state_vec_tmp[j];
	  state_vec[i]=V;
	}

      for (unsigned int j=0; j < aut[0].size(); j++)
	{
	  double vP=0.0;
	  for (unsigned int i=0; i < aut.size(); i++)
	    vP+=pit[i][j]*state_vec[i];
	  future_dist[j]=vP;
	}
      F_Dist.push_back(future_dist);
    }

  return F_Dist;
};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
vector <vector <double> >  PFSA::runX(symbol_list_ history, 
				      unsigned int future_steps,
				      string DISTTYPE)
{  
  vector <vector <double> > F_Dist;
  if (!Xpit.empty())
    if (Xpit.size() == aut.size())
      {
	unsigned int Xalphabet=Xpit[0].size();

	if (PImat.empty())
	  gen_PI();
	if (Gamma.empty())
	  gen_Gamma();

	if (DISTTYPE == "stationary")
	  if (Stationary_distribution.empty())
	    gen_Stationary(10000);
	
	vector <double> Zrow(aut.size(),0.0);
	vector <vector <double> > Z_sq_row(aut.size(),Zrow);
	
	vector <double> state_vec(Zrow);
	state_vec[0]=1.0;
	
	if (DISTTYPE == "stationary")
	  state_vec = Stationary_distribution;
	if  (DISTTYPE == "uniform")
	  state_vec =  vector <double> (aut.size(),1/(aut.size()+0.0));
	if  (DISTTYPE == "random")
	  {
	    state_vec = Zrow;
	    srand ( time(NULL) );
	    if (aut.size() > 1)
	      state_vec[rand()%(aut.size()-1)]=1.0;
	  }

	vector <vector <double> > G(Z_sq_row);
	for (unsigned int i=0;i<aut.size();i++)
	  G[i][i] = 1.0;

	for(size_t sym = 0; sym < history.size(); sym++)
	  {
	    vector <vector <double> > G_next(Gamma[history[sym]]);
	    vector <vector <double> > G_new(aut.size(),Zrow);

	    for(unsigned int i=0;i<aut.size();i++)
	      for(unsigned int k=0;k<aut.size();k++)
		{	      
		  double S=0.0;
		  for(unsigned int j=0;j<aut.size();j++)
		    S+=G[i][j]*G_next[j][k];
		  G_new[i][k]=S;
		}	  
	    for (unsigned int ii=0;ii<aut.size();ii++)
	      for (unsigned int jj=0; jj <aut.size();jj++)
		G[ii][jj] = G_new[ii][jj];
	  }

	vector <double> state_vec_tmp(state_vec);
	for (unsigned int i=0; i < aut.size(); i++)
	  {
	    double V=0.000001;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=G[j][i]*state_vec_tmp[j];
	    state_vec[i]=V;
	  }

	//normalize state_vec
	double S=0.0;
	for (unsigned int i=0;i<aut.size();i++)
	  S+=state_vec[i];
	for (unsigned int i=0;i<aut.size();i++)
	  state_vec[i] /= S;

	//compute future distrubutions
	//first block for step 1
	//second block also propgates the state vec
	vector <double> future_dist(Xalphabet);
	vector <double> fdist_curr(Xalphabet,0.0);
	for (unsigned int i=0; i < Xalphabet; i++)
	  {
	    double V=0.0;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=Xpit[j][i]*state_vec[j];
	    fdist_curr[i]=V;
	  }
	F_Dist.push_back(fdist_curr);

	for (unsigned int f=1;f <future_steps;f++)
	  {
	    vector <double> state_vec_tmp(state_vec);
	    for (unsigned int i=0; i < aut.size(); i++)
	      {
		double V=0.0;
		for (unsigned int j=0; j < aut.size(); j++)
		  V+=PImat[j][i]*state_vec_tmp[j];
		state_vec[i]=V;
	      }

	    for (unsigned int j=0; j < Xalphabet; j++)
	      {
		double vP=0.0;
		for (unsigned int i=0; i < aut.size(); i++)
		  vP+=pit[i][j]*state_vec[i];
		future_dist[j]=vP;
	      }
	    F_Dist.push_back(future_dist);
	  }
      }
  return F_Dist;
};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
vector <vector <double> >  PFSA::runX(symbol_list_ history, 
				      unsigned int future_steps,
				      vector <double> initstate,
				      vector <double> &nextstate)
{  
  vector <vector <double> > F_Dist;
  if (!Xpit.empty())
    if (Xpit.size() == aut.size())
      {
	unsigned int Xalphabet=Xpit[0].size();

	if (PImat.empty())
	  gen_PI();
	if (Gamma.empty())
	  gen_Gamma();

	vector <double> Zrow(aut.size(),0.0);
	vector <vector <double> > Z_sq_row(aut.size(),Zrow);
	
	vector <double> state_vec(initstate);

	vector <vector <double> > G(Z_sq_row);
	for (unsigned int i=0;i<aut.size();i++)
	  G[i][i] = 1.0;

	for(size_t sym = 0; sym < history.size(); sym++)
	  {
	    vector <vector <double> > G_next(Gamma[history[sym]]);
	    vector <vector <double> > G_new(aut.size(),Zrow);

	    for(unsigned int i=0;i<aut.size();i++)
	      for(unsigned int k=0;k<aut.size();k++)
		{	      
		  double S=0.0;
		  for(unsigned int j=0;j<aut.size();j++)
		    S+=G[i][j]*G_next[j][k];
		  G_new[i][k]=S;
		}	  
	    for (unsigned int ii=0;ii<aut.size();ii++)
	      for (unsigned int jj=0; jj <aut.size();jj++)
		G[ii][jj] = G_new[ii][jj];
	  }

	vector <double> state_vec_tmp(state_vec);
	for (unsigned int i=0; i < aut.size(); i++)
	  {
	    double V=0.000001;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=G[j][i]*state_vec_tmp[j];
	    state_vec[i]=V;
	  }

	//normalize state_vec
	double S=0.0;
	for (unsigned int i=0;i<aut.size();i++)
	  S+=state_vec[i];
	if (S > 0.0)
	  for (unsigned int i=0;i<aut.size();i++)
	    state_vec[i] /= S;

	nextstate = state_vec;
	//compute future distrubutions
	//first block for step 1
	//second block also propgates the state vec
	vector <double> future_dist(Xalphabet);
	vector <double> fdist_curr(Xalphabet,0.0);
	for (unsigned int i=0; i < Xalphabet; i++)
	  {
	    double V=0.0;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=Xpit[j][i]*state_vec[j];
	    fdist_curr[i]=V;
	  }
	F_Dist.push_back(fdist_curr);
 
	for (unsigned int f=1;f <future_steps;f++)
	  {
	    vector <double> state_vec_tmp(state_vec);
	    for (unsigned int i=0; i < aut.size(); i++)
	      {
		double V=0.0;
		for (unsigned int j=0; j < aut.size(); j++)
		  V+=PImat[j][i]*state_vec_tmp[j];
		state_vec[i]=V;
	      }

	    for (unsigned int j=0; j < Xalphabet; j++)
	      {
		double vP=0.0;
		for (unsigned int i=0; i < aut.size(); i++)
		  vP+=pit[i][j]*state_vec[i];
		future_dist[j]=vP;
	      }
	    F_Dist.push_back(future_dist);
	  }




      }
  return F_Dist;
};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
vector <vector <double> >  PFSA::runX(symbol_list_ history, 
				      unsigned int future_steps)
{  
  vector <vector <double> > F_Dist;
  if (!Xpit.empty())
    if (Xpit.size() == aut.size())
      {
	unsigned int Xalphabet=Xpit[0].size();

	if (PImat.empty())
	  gen_PI();
	if (Gamma.empty())
	  gen_Gamma();
	if (Stationary_distribution.empty())
	  gen_Stationary();

	vector <double> Zrow(aut.size(),0.0);
	vector <vector <double> > Z_sq_row(aut.size(),Zrow);

	vector <double> state_vec;
	state_vec = Stationary_distribution;

	vector <vector <double> > G(Z_sq_row);
	for (unsigned int i=0;i<aut.size();i++)
	  G[i][i] = 1.0;

	for(size_t sym = 0; sym < history.size(); sym++)
	  {
	    vector <vector <double> > G_next(Gamma[history[sym]]);
	    vector <vector <double> > G_new(aut.size(),Zrow);

	    for(unsigned int i=0;i<aut.size();i++)
	      for(unsigned int k=0;k<aut.size();k++)
		{	      
		  double S=0.0;
		  for(unsigned int j=0;j<aut.size();j++)
		    S+=G[i][j]*G_next[j][k];
		  G_new[i][k]=S;
		}	  
	    for (unsigned int ii=0;ii<aut.size();ii++)
	      for (unsigned int jj=0; jj <aut.size();jj++)
		G[ii][jj] = G_new[ii][jj];
	  }

	vector <double> state_vec_tmp(state_vec);
	for (unsigned int i=0; i < aut.size(); i++)
	  {
	    double V=0.000001;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=G[j][i]*state_vec_tmp[j];
	    state_vec[i]=V;
	  }

	//normalize state_vec
	double S=0.0;
	for (unsigned int i=0;i<aut.size();i++)
	  S+=state_vec[i];
	for (unsigned int i=0;i<aut.size();i++)
	  state_vec[i] /= S;

	//compute future distrubutions
	//first block for step 1
	//second block also propgates the state vec
	vector <double> future_dist(Xalphabet);
	vector <double> fdist_curr(Xalphabet,0.0);
	for (unsigned int i=0; i < Xalphabet; i++)
	  {
	    double V=0.0;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=Xpit[j][i]*state_vec[j];
	    fdist_curr[i]=V;
	  }
	F_Dist.push_back(fdist_curr);

	for (unsigned int f=1;f <future_steps;f++)
	  {
	    vector <double> state_vec_tmp(state_vec);
	    for (unsigned int i=0; i < aut.size(); i++)
	      {
		double V=0.0;
		for (unsigned int j=0; j < aut.size(); j++)
		  V+=PImat[j][i]*state_vec_tmp[j];
		state_vec[i]=V;
	      }

	    for (unsigned int j=0; j < Xalphabet; j++)
	      {
		double vP=0.0;
		for (unsigned int i=0; i < aut.size(); i++)
		  vP+=pit[i][j]*state_vec[i];
		future_dist[j]=vP;
	      }
	    F_Dist.push_back(future_dist);
	  }
      }
  return F_Dist;
};
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
vector <vector <double> >  PFSA::runX(symbol_list_ history, 
				      unsigned int future_steps,
				      state initstate)
{  
  vector <vector <double> > F_Dist;
  if (!Xpit.empty())
    if (Xpit.size() == aut.size())
      {
	unsigned int Xalphabet=Xpit[0].size();

	if (PImat.empty())
	  gen_PI();
	if (Gamma.empty())
	  gen_Gamma();

	vector <double> Zrow(aut.size(),0.0);
	vector <vector <double> > Z_sq_row(aut.size(),Zrow);

	vector <double> state_vec(Zrow);
	state_vec[initstate]=1.0;

	vector <vector <double> > G(Z_sq_row);
	for (unsigned int i=0;i<aut.size();i++)
	  G[i][i] = 1.0;

	for(size_t sym = 0; sym < history.size(); sym++)
	  {
	    vector <vector <double> > G_next(Gamma[history[sym]]);
	    vector <vector <double> > G_new(aut.size(),Zrow);

	    for(unsigned int i=0;i<aut.size();i++)
	      for(unsigned int k=0;k<aut.size();k++)
		{	      
		  double S=0.0;
		  for(unsigned int j=0;j<aut.size();j++)
		    S+=G[i][j]*G_next[j][k];
		  G_new[i][k]=S;
		}	  
	    for (unsigned int ii=0;ii<aut.size();ii++)
	      for (unsigned int jj=0; jj <aut.size();jj++)
		G[ii][jj] = G_new[ii][jj];
	  }

	vector <double> state_vec_tmp(state_vec);
	for (unsigned int i=0; i < aut.size(); i++)
	  {
	    double V=0.000001;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=G[j][i]*state_vec_tmp[j];
	    state_vec[i]=V;
	  }

	//normalize state_vec
	double S=0.0;
	for (unsigned int i=0;i<aut.size();i++)
	  S+=state_vec[i];
	for (unsigned int i=0;i<aut.size();i++)
	  state_vec[i] /= S;

	//compute future distrubutions
	//first block for step 1
	//second block also propgates the state vec
	vector <double> future_dist(Xalphabet);
	vector <double> fdist_curr(Xalphabet,0.0);
	for (unsigned int i=0; i < Xalphabet; i++)
	  {
	    double V=0.0;
	    for (unsigned int j=0; j < aut.size(); j++)
	      V+=Xpit[j][i]*state_vec[j];
	    fdist_curr[i]=V;
	  }
	F_Dist.push_back(fdist_curr);

	for (unsigned int f=1;f <future_steps;f++)
	  {
	    vector <double> state_vec_tmp(state_vec);
	    for (unsigned int i=0; i < aut.size(); i++)
	      {
		double V=0.0;
		for (unsigned int j=0; j < aut.size(); j++)
		  V+=PImat[j][i]*state_vec_tmp[j];
		state_vec[i]=V;
	      }

	    for (unsigned int j=0; j < Xalphabet; j++)
	      {
		double vP=0.0;
		for (unsigned int i=0; i < aut.size(); i++)
		  vP+=pit[i][j]*state_vec[i];
		future_dist[j]=vP;
	      }
	    F_Dist.push_back(future_dist);
	  }
      }
  return F_Dist;
};
//--------------------------------------------------------------------------

Symbolic_string_::Symbolic_string_(symbol_list_ &data_in)
{
  data=symbol_list_ (data_in);
  set <symbol> data_set(data.begin(),data.end());
  alphabet=data_set.size();
}
//--------------------------------------------------------------------------
Symbolic_string_::Symbolic_string_(symbol_list_ &data_in, unsigned int  alph)
{
  data=symbol_list_ (data_in);
  alphabet=alph;
}
//--------------------------------------------------------------------------
Symbolic_string_  Symbolic_string_::operator~()
{
  symbol_list_ ss;
 
  const gsl_rng_type * T;
  gsl_rng * r;      
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());
 
  for(symbol_list_::iterator itr=data.begin(); itr!=data.end();++itr)   
    if ((*itr)==gsl_rng_uniform_int(r, alphabet))
      ss.push_back(*itr);
    
  gsl_rng_free (r);
  
  return Symbolic_string_(ss,alphabet);
}

//--------------------------------------------------------------------------

Symbolic_string_::Symbolic_string_(Symbolic_string_ const &S)
{
  data=S.data;
  //  set <symbol> data_set(data.begin(),data.end());
  alphabet=S.alphabet;
}
//--------------------------------------------------------------------------

Symbolic_string_  Symbolic_string_::operator+(const Symbolic_string_ &S)
{
  symbol_list_ ss;
  symbol_list_::const_iterator itrB=S.data.begin();
  for(symbol_list_::iterator itrA=data.begin(); 
      itrA!=data.end() && itrB!=S.data.end();++itrA)
    if (*itrA==*(itrB++))
      ss.push_back(*itrA);

  return Symbolic_string_(ss,alphabet);
}  /*<!-- add symbolic streams -->*/

//--------------------------------------------------------------------------
Symbolic_string_  Symbolic_string_::operator+(const symbol_list_ &S)
{
  symbol_list_ ss;
  symbol_list_::const_iterator itrB=S.begin();
  for(symbol_list_::iterator itrA=data.begin(); 
      itrA!=data.end() && itrB!=S.end();++itrA)
    if (*itrA==*(itrB++))
      ss.push_back(*itrA);
  return Symbolic_string_(ss,alphabet);
}  /*<!-- add symbolic streams -->*/
//--------------------------------------------------------------------------
//------------------------------------
Symbolic_string_  Symbolic_string_::operator*(PFSA &G)
{
  //lock_guard<mutex> guard(mtx_);
  const gsl_rng_type * T;
  gsl_rng * r;   /* create  generator chosen by  env  var GSL_RNG_TYPE */   
  gsl_rng_env_setup(); 
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, random_seed());
  /** \b Random initial state */
  srand ( time(NULL) );
  
  connx aut = G.get_aut();
  pitilde pit = G.get_pit();
  unsigned int alphabet_B=pit[0].size();

  int curr_state;
  if (aut.size() > 1)
    curr_state=rand()%(aut.size()-1);
  else
    curr_state=0;
  unsigned int len = data.size();
  /*  if ((GLOBAL_VAR_DATA_LEN_SET=true) 
      && (GLOBAL_VAR_DATA_LEN != -1))
    len = GLOBAL_VAR_DATA_LEN;
  */
  symbol_list_ s(len);
  for(size_t i = 0; i < len; i++)
    {
      if (aut[curr_state][data[i]] != -1)
	{
      double cumprob=0.0;
      double key = gsl_ran_flat (r, 0.0, 1.0);
      symbol sym;
      sym=0;

      for (sym=0; sym < alphabet_B-1; sym++)
	{
	  cumprob = cumprob + pit[curr_state][sym];
	  if (key <= cumprob)
	    break;
	}
      s[i] = sym;
      curr_state=aut[curr_state][data[i]];
	}
    }
  gsl_rng_free (r);
  return Symbolic_string_(s,alphabet_B);
};
//-------------------------------------------------------------
//--------------------------------------------------------------------------
Symbolic_string_   Symbolic_string_::operator!()
{
  vector <Symbolic_string_ > Ws;
  symbol_list_ ss;

  for (symbol  alph(0); alph < alphabet-1; alph++)
    Ws.push_back(~(*this));


  unsigned int len=Ws[0].data.size();
  for (unsigned int i=1; i<Ws.size();i++)
    if (Ws[i].data.size() < len)
      len=Ws[i].data.size();

  set <symbol> alph_set0;
  for (symbol i(0); i<alphabet;i++)
    alph_set0.insert(i);

  for (unsigned int i=0;i<len;i++)
    {
      set <symbol> alph_set(alph_set0);
      /*    for (unsigned int  j=0; j<(unsigned int) alphabet-1;j++)
	    {
	    alph_set.erase(*(Ws[j].data.begin()));
	    Ws[j].data.erase(Ws[j].data.begin());
	    }
      */
      for (unsigned int  j=0; j<(unsigned int) alphabet-1;j++)
	alph_set.erase((Ws[j]).data[i]);
      
      if (alph_set.size() == 1)
	ss.push_back(*(alph_set.begin()));
    }
  // }
  return Symbolic_string_(ss,alphabet);
}
//--------------------------------------------------------------------------
symbol_list_ reflect(symbol_list_ S, unsigned int alphabet)
{
  symbol_list_ ss;

  for(unsigned int i=0; i < S.size();++i)
    {
      symbol sym(S[i]+1);
      if (sym == alphabet)
	sym = 0;
      ss.push_back(sym);
    }
  return ss;
}

//--------------------------------------------------------------------------
Symbolic_string_::Symbolic_string_(vector < double > cdata,vector < double > partition)
{
  cont_dom_partition=partition;
  for (unsigned int i=0; i < cdata.size(); i++)
    {
      symbol j(0);
      for (j=0; j<cont_dom_partition.size();j++)
	if (cdata[i]<cont_dom_partition[j])
	  break;
      data.push_back(j);
    }
  alphabet=(symbol)partition.size()+1;
}
//--------------------------------------------------------------------------
Symbolic_string_::Symbolic_string_(string sdata_file)
{
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
  symbol sym;
   

  while (getline (DATA, line))
    {
      ss.clear();
      ss.str ("");
      ss << line;
      ss >> sym;
      data.push_back(sym);       
    }
  set <symbol> data_set(data.begin(),data.end());
  alphabet=(symbol)data_set.size();
}
//--------------------------------------------------------------------------
void  get_Symbolic_DataMatrix(string sdata_file, 
			      char sep,
			      unsigned int SZ,
			      vector < symbol_list_ >* v_sym)
{
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
  symbol sym;
   
  // vector <Symbolic_string_> v_sym;
  while (getline (DATA, line))
    {
      symbol_list_ data;
      ss.clear();
      ss.str ("");
      ss << line;
      while (data.size() < SZ && ss.good())
	{
	  ss >> sym;
	  data.push_back(sym);
	  ss >> sep;
	}
      v_sym->push_back(data);
    }
}
//--------------------------------------------------------------------------
void  get_Symbolic_DataMatrix(string sdata_file, 
			      char sep,
			      vector < symbol_list_ >* v_sym)
{
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
  symbol sym;
   
  // vector <Symbolic_string_> v_sym;
  while (getline (DATA, line))
    {
      symbol_list_ data;
      ss.clear();
      ss.str ("");
      ss << line;
      while (ss.good())
	{
	  ss >> sym;
	  data.push_back(sym);
	  ss >> sep;
	}
      //       Symbolic_string_ sym_tmp(data);
      v_sym->push_back(data);
      //       return v_sym;
    }
}
//--------------------------------------------------------------------------
/**
   Specify continuous data file with each line in the file 
   corresponding to each run of possibly different underlying generators.
   The partition is specfified, and we read in line by line, pass it through the
   partition and get a vector of vector of symbols
*/
//--------------------------------------------------------------------------
void  get_continuous_DataMatrix(string cdata_file, 
				vector < double > partition,
				unsigned int SZ,
				vector < symbol_list_ >* v_sym)
{
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;

  while (getline (DATA, line))
    {
      symbol_list_ data;
      ss.clear();
      ss.str ("");
      ss << line;
      while (data.size() < SZ && ss.good())
	{
	  ss >> data_pt;
	  symbol j(0);
	  for (j=0; j<partition.size();j++)
	    if (data_pt<partition[j])
	      break;
	  data.push_back(j);
	}
      v_sym->push_back(data);
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------
void  get_continuous_DataMatrix(unsigned int begnum, 
				unsigned int endnum, 
				unsigned int fixedwidth,
				vector <unsigned int> columnindices,
				matrix_dbl partition,
				unsigned int SZ,
				vector < Symbolic_string_ >& v_sym)
{
  // ensure columnindices are in ascending order, and find the max
  sort (columnindices.begin(), columnindices.end());

  for (unsigned int file_index=begnum; file_index<=endnum; file_index++)
    {
      char cdata_file[1024], format[128];

      sprintf(format,"%s%dd%s","%0",fixedwidth,".txt");
      sprintf(cdata_file,format, file_index);
      ifstream DATA(cdata_file);
      string line;
      stringstream ss;
      symbol_list_ data;
      while (data.size() < SZ && getline (DATA, line))
	{
	  symbol data_pt(0);

	  ss.clear();
	  ss.str ("");
	  ss << line;

	  double tmp;	  
	  vector <double> sym_vec;

	  unsigned int colnum;
	  unsigned int colnum_last=0;

	  for (unsigned int colcounter=0; colcounter<columnindices.size(); colcounter++)
	    {
	      colnum=columnindices[colcounter];
	      for (unsigned int dummyc=colnum_last; dummyc <= colnum; dummyc++)
		if (ss.good())
		  ss >> tmp;
		
	      sym_vec.push_back(tmp);
	      colnum_last=colnum+1;
	    }

	  if (!(sym_vec.size() <= columnindices.size()))
	    cout << "Error In Reading Multiple Column" << endl;

	  for (unsigned int i=0; i < columnindices.size();i++)
	    {
	      symbol j(0);
	      for (j=0; j<partition[i].size();j++)
		if (sym_vec[i]<partition[i][j])
		  break;
	      data_pt += j*pow(partition[i].size()+1,i);
	    }
	  data.push_back(data_pt);
	}
      symbol alphabet(1);
      for (unsigned int i=0; i < columnindices.size();i++)
	alphabet *= partition[i].size()+1;

      Symbolic_string_ tmp(data,alphabet);
      v_sym.push_back(tmp);
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------
void  get_continuous_DataMatrix(unsigned int begnum, 
				unsigned int endnum, 
				unsigned int fixedwidth,
				unsigned int columnnum,
				vector < double > partition,
				unsigned int SZ,
				vector < Symbolic_string_ >& v_sym)
{
  for (unsigned int file_index=begnum; file_index<=endnum; file_index++)
    {
      symbol alphabet(partition.size()+1);
      char cdata_file[1024], format[128];

      sprintf(format,"%s%dd%s","%0",fixedwidth,".txt");
      sprintf(cdata_file,format, file_index);
      ifstream DATA(cdata_file);
      string line;
      stringstream ss;
      double data_pt;
      symbol_list_ data;
      while (data.size() < SZ && getline (DATA, line))
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  double tmp;	  
	  for (unsigned int skipcol=0; skipcol<columnnum; skipcol++)
	    ss >> tmp;

	  if (ss.good())
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break;
	      data.push_back(j);
	    }
	  else
	    {
	      cout << " Column number too large for datafile " << endl;
	      return;
	    }
	}
      Symbolic_string_ tmp(data,alphabet);
      v_sym.push_back(tmp);
    }
}
//--------------------------------------------------------------------------

//--------------------------------------------------------------------

void  get_continuous_DataMatrix(unsigned int begnum, 
				unsigned int endnum, 
				unsigned int fixedwidth,
				unsigned int columnnum,
				vector < double > partition,
				unsigned int SZ,
				vector < symbol_list_ >* v_sym)
{
  for (unsigned int file_index=begnum; file_index<=endnum; file_index++)
    {


  
      char cdata_file[1024], format[128];
      sprintf(format,"%s%dd%s","%0",fixedwidth,".txt");
      sprintf(cdata_file,format, file_index);
      //      cout << cdata_file << endl;
      ifstream DATA(cdata_file);
      string line;
      stringstream ss;
      double data_pt;
      symbol_list_ data;
      while (data.size() < SZ && getline (DATA, line))
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  double tmp;	  
	  for (unsigned int skipcol=0; skipcol<columnnum; skipcol++)
	    ss >> tmp;

	  if (ss.good())
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break;
	      data.push_back(j);
	    }
	  else
	    {
	      cout << " Column number too large for datafile " << endl;
	      return;
	    }
	}
      v_sym->push_back(data);
    }
}
//--------------------------------------------------------------------
void  get_continuous_DataMatrix(string cdata_file,
				vector < double > partition,
				unsigned int SZ,
				vector < Symbolic_string_ >& v_sym)
{
  symbol alphabet(partition.size()+1);
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;
  symbol_list_ data;
  while (data.size() < SZ && getline (DATA, line))
    {
      ss.clear();
      ss.str ("");
      ss << line;
      while (ss.good())
	{
	  ss >> data_pt;
	  symbol j(0);
	  for (j=0; j<partition.size();j++)
	    if (data_pt<partition[j])
	      break; 
	  data.push_back(j);
	}
      Symbolic_string_ tmp(data,alphabet);
      v_sym.push_back(tmp);
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------
void  get_continuous_DataMatrix(string cdata_file,
				string data_dir,
				vector <double> partition,
				unsigned int SZ,
				symbol_list_ &data)
{
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;
  if (data_dir == "across")
    {
      if (getline (DATA, line))
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  while (ss.good() && data.size() < SZ)
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data.push_back(j);
	    }
	}
      else
	cout << "COULDNOT READ DATAFILE" << endl;
    }
  else
    {
      while (getline (DATA, line) && data.size() < SZ)
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  if (ss.good())
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data.push_back(j);
	    }
	}
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
/**
   Get continuous data, pass through specified partition, and get symbol string
*/
//--------------------------------------------------------------------------
Symbolic_string_::Symbolic_string_(string cdata_file,vector < double > partition)
{
  cont_dom_partition=partition;

  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;

  while (getline (DATA, line))
    {
      ss.clear();
      ss.str ("");
      ss << line;
      ss >> data_pt;

      symbol j(0);
      for (j=0; j<cont_dom_partition.size();j++)
	if (data_pt<cont_dom_partition[j])
	  break;
      data.push_back(j);
    }
  alphabet=(symbol)partition.size()+1;
  /*
    set <symbol> data_set(data.begin(),data.end());
    alphabet=data_set.size();
  */
}
//--------------------------------------------------------------------------

Symbolic_string_::Symbolic_string_(string cdata_file,
				   vector < double > partitionX,
				   vector <double> partitionY)
{
  //cont_dom_partition=partition;

  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_ptX, data_ptY;

  while (getline (DATA, line))
    {
      ss.clear();
      ss.str ("");
      ss << line;
      ss >> data_ptX;
      ss >> data_ptY;

      unsigned int k=0,j=0;
      for (j=0; j<partitionX.size();j++)
	if (data_ptX<partitionX[j])
	  break;
      for (k=0; k<partitionY.size();k++)
	if (data_ptY<partitionY[k])
	  break;
      symbol sym(j+k*partitionY.size());
      data.push_back(sym);
    }
  set <symbol> data_set(data.begin(),data.end());
  alphabet=(symbol)data_set.size();
}


//--------------------------------------------------------------------------
void Symbolic_string_::get_norm()
{
  if (data.size()==0)
    norm=10;
  else
    {
      vector <double> sfreq((size_t)alphabet,0.0);
      double S=0.0;
      for (symbol_list_::iterator itr=data.begin();
	   itr != data.end();
	   ++itr)
	sfreq[*itr]++;
      for (symbol  i(0); i <alphabet; i++)
	S+=(sfreq[i]/(data.size()+0.0)-1/((double)alphabet+0.0))
	  *(sfreq[i]/(data.size()+0.0)-1/((double)alphabet+0.0));
      norm=sqrt(S);
    }
}

//------------------------------------
symbol_list_ & Symbolic_string_::inc(symbol_list_ &s)
{
  if (s.empty())
    {
      s.push_back(symbol (0) );
      return s;
    }
  symbol sigma=s.back();
  s.pop_back();

  if (sigma++ < alphabet-1)
    s.push_back(sigma);
  else
    {
      s = inc(s);
      s.push_back(symbol (0));
    }
  return s;
}

//--------------------------------------------------------------------------
void Symbolic_string_::get_reflected_norm(unsigned int DEPTH)
{
  symbol_list_ omega(DEPTH, symbol (0));
  while (omega.size() == DEPTH)
    {
      gen_Phi(omega);
      inc(omega);
    }
  for(phi_data_type_::iterator itr=Phi.begin();
      itr!=Phi.end();
      ++itr)
    {
      symbol_list_ omega1=reflect(itr->first, alphabet);
      vector <unsigned int>  v1,v2;
      for (symbol i(0); i < alphabet;i++)
	{
	  v1.push_back((itr->second)[i]);
	  v2.push_back(Phi[omega1][i]);
	}
      map < symbol , unsigned int > v;

      for (symbol i(0); i < alphabet;i++)
	v[i] = v1[i]+v2[i];
      Phi[omega]=v;
      Phi[itr->first]=v;
    }
  symbol_list_ lambda;
  norm=0.0;
  vector <double> U(alphabet,1/(alphabet+0.0));

  //cout << dec<<  Phi << endl;
  for(phi_data_type_::iterator itr=Phi.begin();
      itr!=Phi.end();
      ++itr)
    {
      unsigned int str_size = itr->first.size();
      vector <double> symb_deriv(mk_stochastic(itr->second));
      double inf_norm=0.0;

      for (unsigned int sym=0; sym < alphabet; ++sym)
	if (fabs(U[sym]-symb_deriv[sym]) > inf_norm)
	  inf_norm = fabs(U[sym]-symb_deriv[sym]);
 
      norm+=inf_norm/pow(alphabet+0.0,2*str_size+0.0);
    }

  norm *= (alphabet-1)/(alphabet+0.0);
}

//--------------------------------------------------------------------------
void Symbolic_string_::get_norm(unsigned int DEPTH)
{
  symbol_list_ omega(DEPTH,symbol (0));
  while (omega.size() == DEPTH)
    {
      gen_Phi(omega);
      inc(omega);
    }
  symbol_list_ lambda;
  norm=0.0;
  vector <double> U(alphabet,1/(alphabet+0.0));

  //cout << dec<<  Phi << endl;
  for(phi_data_type_::iterator itr=Phi.begin();
      itr!=Phi.end();
      ++itr)
    {
      unsigned int str_size = itr->first.size();
      vector <double> symb_deriv(mk_stochastic(itr->second));
      if (symb_deriv.empty())
	symb_deriv = U;

      double inf_norm=0.0;

      for (unsigned int sym=0; sym < alphabet; ++sym)
	if (fabs(U[sym]-symb_deriv[sym]) > inf_norm)
	  inf_norm = fabs(U[sym]-symb_deriv[sym]);
 
      norm+=inf_norm/pow(alphabet+0.0,2*str_size+0.0);
    }
  norm *= (alphabet-1)/(alphabet+0.0);
}

void Symbolic_string_::get_norm_new(unsigned int DEPTH)
{
	typedef map<unsigned int, vector<unsigned int>> DataPhi;
	vector<unsigned int> init;
	for (unsigned int i = 0; i < data.size(); i++)
	{
		init.push_back(i);
	}
	DataPhi odp;
	odp[0] = init;
	
	norm = 0;
	for (unsigned int l = 0; l <= DEPTH; l++)
	{
		DataPhi ndp;
		DataPhi::iterator itr;
		double normL = 0;
		for (itr = odp.begin(); itr != odp.end(); ++itr)
		{
			unsigned int total = 0;
			vector<double> prob(alphabet, 0);
			unsigned int wordSig = itr->first;
			for (unsigned int index : itr->second)
			{
				unsigned int i = index + l;
				if (i < data.size())
				{
					ndp[wordSig * alphabet + data[i]].push_back(index);
					prob[data[i]] += 1.0;
					total += 1;
				}
			}
			if (total > 0)
			{
				double max_diff = 0;
				for (double p : prob)
				{
					double diff = abs(p / total - 1.0 / alphabet);
					max_diff = (max_diff > diff) ? max_diff : diff;
				}
				normL += max_diff;
			}
		}
		norm += normL / pow((double)alphabet, 2.0 * l);
		odp = ndp;
	}
	norm *= (alphabet - 1.0) / alphabet; 
}
//--------------------------------------------------------------------------

void Symbolic_string_::gen_Phi(symbol_list_ omega)
{
  if (!Phi[omega].empty())
    return;
  if (omega.empty())
    {
      /**	 Treat empty string separately      */
      for(unsigned int i=0; 
	  i <data.size();
	  ++i)
	{
	  Phi[omega][data[i]]++;
	  Datapin[omega][data[i]].push_back(i);
	}
      return;
    }

  unsigned int omega_Len=omega.size();
  symbol sym=omega.back();
  symbol_list_ omega0(omega);
  omega0.pop_back();

  /**	 Recursive call to getPhi()      */ 
  if (Phi[omega0].empty())
    gen_Phi(omega0); 

  for(vector < unsigned int >::iterator itr
	=Datapin[omega0][sym].begin();
      itr!=Datapin[omega0][sym].end();
      ++itr)
    if (*itr+omega_Len < data.size())
      {
	Phi[omega][data[*itr+omega_Len]]++;
	Datapin[omega][data[*itr+omega_Len]].push_back(*itr);
      }
  for (map <symbol, unsigned int>::iterator itr=Phi[omega0].begin();
       itr!=Phi[omega0].end();
       ++itr)
    {    
      map <symbol, unsigned int>::iterator itr1=Phi[omega].find(itr->first);
      if (itr1 == Phi[omega].end())
	Phi[omega][itr->first]=0;
    }
}
//-----------------------------------------------------------

bool Symbolic_string_::gen_Phi_FAST(symbol_list_ omega)
{
  if (!Phi[omega].empty())
    return true;
  if (omega.empty())
    {
      /**	 Treat empty string separately      */
      for(unsigned int i=0; 
	  i <data.size();
	  ++i)
	{
	  Phi[omega][data[i]]++;
	  Datapin[omega][data[i]].push_back(i);
	}
      return true;
    }

  unsigned int omega_Len=omega.size();
  symbol sym=omega.back();
  symbol_list_ omega0(omega);
  omega0.pop_back();

  /**	 Recursive call to getPhi()      */ 
  if (Phi[omega0].empty())
    return false;

  // ensures that we retuen false if the derivative is going to be all zeros //
  if (Datapin[omega0][sym].empty())
    return false;

  for(vector < unsigned int >::iterator itr
	=Datapin[omega0][sym].begin();
      itr!=Datapin[omega0][sym].end();
      ++itr)
    if (*itr+omega_Len < data.size())
      {
	Phi[omega][data[*itr+omega_Len]]++;
	Datapin[omega][data[*itr+omega_Len]].push_back(*itr);
      }
  for (map <symbol, unsigned int>::iterator itr=Phi[omega0].begin();
       itr!=Phi[omega0].end();
       ++itr)
    {    
      map <symbol, unsigned int>::iterator itr1=Phi[omega].find(itr->first);
      if (itr1 == Phi[omega].end())
	Phi[omega][itr->first]=0;
    }

  // Implementation of forward slip stop //
  /* if (omega_Len > 2)
    {
      if (!zero(Phi[symbol_list_ (omega.begin()+1,omega.end())]))
	{
	  if (dist_entropy( mk_stochastic(Phi[symbol_list_ (omega.begin()+1,omega.end())])) 
	      <= dist_entropy( mk_stochastic(Phi[omega])))
	    {
	      // roll back changes to Phi and Datapin
	      Phi.erase(omega);
	      Datapin.erase(omega);
	      return false;
	    }
	}
    }
  */

  //   symbol_list_ omega_H = omega_syn;
  //   copy(omega.begin()+1+omega_syn.size(),omega.end(),back_inserter(omega_H));








  /*
      symbol_list_ omega_H(omega.begin()+1, omega.end());
      if (zero(Phi[omega_H]))
	gen_Phi(omega_H);
      

      //cout << omega << "  <--|-->  " << omega_H << endl;

      if (dist_entropy( mk_stochastic(Phi[omega_H])) 
	  <= dist_entropy( mk_stochastic(Phi[omega])))
	{
	  // roll back changes to Phi and Datapin
	  Phi.erase(omega);
	  Datapin.erase(omega);
	  return false;
	}
  */



  return true;
}
//-----------------------------------------------------------
//------------------------------------
symbol_list_ &Symbolic_string_::getSync(double MIN_DEV_SYNC_SEARCH_,
					unsigned int SYN_PREF_LENGTH_)
{
  unsigned int OCC_STOP=2;
  Phi.clear();
  bool SYNC=false;
  symbol_list_ lambda;
  unsigned int sync_pref_len=0;


  // cout << "xxx " << SYN_PREF_LENGTH_<<  endl;
  while ( sync_pref_len++ < SYN_PREF_LENGTH_)
    {
      gen_Phi(lambda);
      vector <double> v0 = mk_stochastic(Phi[lambda]);
		
      double distance=0.0;
      for(symbol sigma(0); sigma < alphabet;++sigma)
	{
	  symbol_list_ omega_H;
	  omega_H.push_back(sigma);
	  copy(lambda.begin(),lambda.end(),back_inserter(omega_H));
	  gen_Phi(omega_H);
			
	  vector <double> v1;
	  if (zero(Phi[omega_H]))
	    {
	      if (lambda.size() < OCC_STOP)
		{
		  SYNC=true;
		  break;
		}
	      v1 = v0;
	    }
	  else
	    v1 = mk_stochastic(Phi[omega_H]);
			
	  for(unsigned int i=0; i < alphabet;++i)
	    distance += fabs(v0[i]-v1[i]);		
	}
      distance /= (alphabet*alphabet+0.0);

      //cout << distance << " " ;

      if (distance <= MIN_DEV_SYNC_SEARCH_)
	{
	  SYNC=true;
	  break;
	}
      else
	inc(lambda);
    }
  if (SYNC)
    omega_syn=lambda;

  //cout << "syn " << omega_syn << endl;

  return omega_syn;
}
//------------------------------------
void Symbolic_string_::getSync()
{
  unsigned int OCC_STOP=2;
  if (omega_syn.empty())
    {
      unsigned int SYN_PREF_LENGTH_=10000;
      double MIN_DEV_SYNC_SEARCH_=0.01;

      Phi.clear();
      bool SYNC=false;
      symbol_list_ lambda;
      unsigned int sync_pref_len=0;

      while ( sync_pref_len++ < SYN_PREF_LENGTH_)
	{

	  //cout << lambda << endl;

	  gen_Phi(lambda);
	  vector <double> v0 = mk_stochastic(Phi[lambda]);
		
	  double distance=0.0;
	  for(symbol sigma(0); sigma < alphabet;++sigma)
	    {
	      symbol_list_ omega_H;
	      omega_H.push_back(sigma);
	      copy(lambda.begin(),lambda.end(),back_inserter(omega_H));
	      gen_Phi(omega_H);
			

	      vector <double> v1;
	      if (zero(Phi[omega_H]))
		{
		  if (lambda.size() < OCC_STOP)
		    {
		      SYNC=true;
		      break;
		    }
		  v1 = v0;
		}
	      else
		v1 = mk_stochastic(Phi[omega_H]);
			
	      for(unsigned int i=0; i < alphabet;++i)
		distance += fabs(v0[i]-v1[i]);		
	    }
	  distance /= (alphabet*alphabet+0.0);
	  if (distance <= MIN_DEV_SYNC_SEARCH_)
	    {
	      SYNC=true;
	      break;
	    }
	  else
	    inc(lambda);
	}
      if (SYNC)
	omega_syn=lambda;
    }
}
//------------------------------------
//------------------------------------
double Symbolic_string_::entropy()
{
  getSync();
  symbol_list_ omega(omega_syn);
  unsigned int MAX_DEPTH=10;

  stoch_phi_data_type_ sPhi;
  map <symbol_list_, double> occ_list;

  phi_data_type_ Phi_syncpref;
  double Val=0.0;
  unsigned int loopcount=0;

  while (1)
    {
      loopcount++;
      gen_Phi(omega);
      double val=0.0;
      for (symbol sym(0); sym < alphabet; sym++)
	val += Phi[omega][sym];
      /*
      if (val == 0)
	break;
      */
      if (loopcount > MAX_DEPTH)
	break;
 
      Phi_syncpref[omega] = Phi[omega];
      occ_list[omega]=val;
      Val += val;
      omega = inc(omega);
    }

  if (Val > 0.0)
    for (map <symbol_list_, double>::iterator itr=occ_list.begin();
	 itr!=occ_list.end();
	 ++itr)
      itr->second /= Val;

  mk_stochastic(Phi_syncpref,sPhi);

  double entropy=0.0;

  for(stoch_phi_data_type_::iterator itr=sPhi.begin();
      itr!=sPhi.end();
      ++itr)
    {
      double e_tmp=0.0;
      for (unsigned int i=0; i < (unsigned int)alphabet; ++i)
	{
	  double p=(itr->second)[i];
	  if (p>0.0)
	    e_tmp -= p*log(p);
	}
      entropy += e_tmp*occ_list[itr->first];
    }
  entropy /= log(2);
  return entropy;
}
//------------------------------------
//------------------------------------
void Symbolic_string_::Phi_clean(unsigned int slen)
{
  for(phi_data_type_::iterator itr=Phi.begin();
      itr!=Phi.end();
      ++itr)
    if (itr->first.size()>slen)
      {
	Phi.erase(Phi.begin(),itr);
	break;
      }
}
//------------------------------------
void Symbolic_string_::Datapin_clean(unsigned int slen)
{
  for(pin_data_type_::iterator itr=Datapin.begin();
      itr!=Datapin.end();
      ++itr)
    if (itr->first.size()>slen)
      {
	Datapin.erase(Datapin.begin(),itr);
	break;
      }
}
//------------------------------------
double Symbolic_string_::entropy(unsigned int MAX_DEPTH, double epsilon, int MIN_OCC)
{
  //getSync();
  //symbol_list_ omega(omega_syn);  
  map < vector <double> , 
	double, 
	eps_compare_ > phiMap(eps_compare_((const double)epsilon));
	    
  unsigned int loopcount=0;
  double numEntries=0.0;
  gen_Phi(omega_syn);

  symbol_list_ omega;
  while (1)
    {
      symbol_list_ omega_full(omega_syn);
      copy (omega.begin(),omega.end(),back_inserter(omega_full));
      
      if (gen_Phi_FAST(omega_full))
	{
	  
	  double val=0.0;
	  for (symbol sym(0); sym < alphabet; sym++)
	    val += Phi[omega_full][sym];
	  if (val>MIN_OCC)
	    {
	      /**/
		phiMap[mk_stochastic(Phi[omega_full])]+=val;
		numEntries+=val;
		/*  */
	      
	      /* alternate state prob computation */
	      //cout << omega_full << " --- " << Phi[omega_full] << " ... " << val << endl;
	      
	      //phiMap[mk_stochastic(Phi[omega_full])]++;
	      //numEntries++;

	      /*   */
	    }
	}

      unsigned int cur_omega_len=omega_full.size();
      omega = inc(omega);
      unsigned int next_omega_len=omega.size()+omega_syn.size();
      if ((next_omega_len > cur_omega_len) 
	  && (cur_omega_len >1))
	{
	  Phi_clean(cur_omega_len-1);
	  Datapin_clean(cur_omega_len-1);
	}

      if (loopcount++ > MAX_DEPTH)
	break;
    }

  if (numEntries > 0.0)
    for (map < vector <double> , 
	   double, 
	   eps_compare_ >::iterator itr=phiMap.begin();
	 itr!=phiMap.end();
	 ++itr)
      itr->second /= numEntries;

  double entropy=0.0;
  for(map < vector <double> , 
	double, 
	eps_compare_ >::iterator itr=phiMap.begin();
      itr!=phiMap.end();
      ++itr)
    {
      double e_tmp=0.0;
      for (unsigned int i=0; i < (unsigned int)alphabet; ++i)
	{
	  //cout << itr->first << endl;
	  double p=(itr->first)[i];
	  if (p>0.0)
	    e_tmp -= p*log(p);
	}
      entropy += e_tmp*(itr->second);
    }
  entropy /= log(2);
  return entropy;
}
//------------------------------------
void drawGraph(connx aut, pitilde pit, string dotfile, string Type)
{
  symbol alphabet(aut[0].size());
  unsigned int NUM_STATE=aut.size();

  Digraph g(NUM_STATE);

  vector < string >  name_string_arr;
  string  name[NUM_STATE];
  for (unsigned int i = 0; i < NUM_STATE; i++)
    {
      char buff[2048];
      sprintf(buff,"<q<SUB>%d</SUB>>",i);//<Regular<SUB>subscript</SUB>>
      name[i] =  buff;
    } 

  property_map<Digraph, edge_name_t>::type namemap = get(edge_name, g);
  int edgecount=0;
  for (unsigned int num_state=0;num_state<NUM_STATE; num_state++)
    for (symbol s(0); s < alphabet; s++)
      if (aut[num_state][s] != -1)
	{
	  char buff1[2048],buff2[1024];
	  sprintf(buff1,"%d",(int)s);
	  sprintf(buff2,"(%1.2f)",pit[num_state][s]);
	  string strtmp ="";
	  strtmp += buff1;
	  strtmp += buff2;
	  name_string_arr.push_back(strtmp);	  
	  graph_traits<Digraph>::edge_descriptor e; 
	  bool inserted;
	  tie(e, inserted) = add_edge( num_state,  aut[num_state][s], g);
	  namemap[e] = name_string_arr[edgecount++];
	}

  boost::property_map<Digraph, edge_name_t>::type  edge_name_arr = get(edge_name, g);
  std::map<std::string,std::string> graph_attr, vertex_attr, edge_attr;


  string graphsize = "40,25", 
    rankdir = "TD", 
    nodeshape = "circle",
    DPI="600",
    nodestyle = "filled", 
    nodecolor = "powderblue", 
    nodefontsize="16",
    edgeweight = "20",
    edgefontcolor = "midnightblue", 
    edgefontweight = "bold",
    nodefontweight = "bold",
    nodefontcolor="black",
    edgestyle = "solid", 
    edgecolor = "slateblue1", 
    edgefontname = "courier",
    nodefontname = "courier",
    nodefillcolor = "mintcream", 
    graphname = "PFSA",
    dotprog="dot",
    edgepenwidth="1",
    arrowsize="1";
  
  CONFIG cfg("dot.cfg");
  cfg.set(graphsize,"GRAPH_SIZE");
  cfg.set(rankdir,"RANKDIR");
  cfg.set(DPI,"DPI");
  cfg.set(nodestyle,"NODESTYLE");
  cfg.set(nodecolor,"NODECOLOR");
  cfg.set(nodefontsize,"NODEFONTSIZE");
  cfg.set(nodefontweight,"NODE_FONTWEIGHT");
  cfg.set(edgeweight,"EDGE_WEIGHT");
  cfg.set(edgefontcolor,"EDGE_FONTCOLOR");
  cfg.set(nodefontweight,"EDEGE_FONTWEIGHT");
  cfg.set(nodefontcolor,"NODE_FONTCOLOR");
  cfg.set(edgestyle,"EDGE_STYLE");
  cfg.set(edgefontname,"EDGE_FONTNAME");
  cfg.set(nodefontname,"NODE_FONTNAME");
  cfg.set(edgecolor,"EDGE_COLOR");
  cfg.set(edgepenwidth,"EDGE_PENWIDTH");
  cfg.set(nodefillcolor,"NODE_FILLCOLOR");
  cfg.set(graphname,"GRAPH_NAME");
  cfg.set(arrowsize,"ARROW_SIZE");
  cfg.set(Type,"GRAPH_TYPE");
  cfg.set(dotprog,"DOT_PROG");

  string Str1=dotprog + " -T";
  string DCOM;

  graph_attr["bgcolor"] = "transparent";
  graph_attr["size"] = graphsize;
  graph_attr["rankdir"] = rankdir;
  graph_attr["dpi"] = DPI;
  //	  graph_attr["ratio"] = "fill";
  vertex_attr["shape"] = nodeshape;
  vertex_attr["style"] = nodestyle;
  vertex_attr["color"] = nodecolor;//"indianred";
  vertex_attr["fillcolor"] = nodefillcolor;//"indianred";
  vertex_attr["fontcolor"] = nodefontcolor;//"indianred";
  vertex_attr["fontweight"] = nodefontweight;//"indianred";
  vertex_attr["fontsize"] = nodefontsize;//"indianred";
  vertex_attr["fontname"] = nodefontname;//"indianred";
  edge_attr["fontcolor"] = edgefontcolor;
  edge_attr["fontname"] = edgefontname;
  edge_attr["fontweight"] = edgefontweight;
  edge_attr["weight"] = edgeweight;
  edge_attr["style"] = edgestyle;
  edge_attr["color"] = edgecolor;
  edge_attr["penwidth"] = edgepenwidth;
  edge_attr["arrowsize"] = arrowsize;

  
  ofstream outfile((dotfile+".dot").c_str());
  write_graphviz(outfile, g,	
		 make_label_writer_html(name), make_label_writer(edge_name_arr),
		 make_graph_attributes_writer(graph_attr, vertex_attr, 
					      edge_attr ));
  outfile.close();
  if (Type != "NO_DOT")
    {
      DCOM=Str1+Type+ " " + dotfile +".dot -o"+ " " + dotfile+"."+Type; 
      std::system(DCOM.c_str());
    }
}
//--------------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S)
{
  TOLERANCE=0.000001;
  num_elements=S.size();

  /** 
      Create distance matrix dMap 
      via semantic annihilation
  */
  for (unsigned int i=0;i<num_elements;i++)
    for (unsigned int j=0;j<num_elements;j++)
      {
	Symbolic_string_ tmp(S[i] + !S[j]);
	tmp.get_norm();
	dMap[i][j] = tmp.norm;
      }

  /**
     Create DMap_tmp as the 
     zero-diagonal symmetricized version 
     of the distance matrix dMap;
  */
  matrix_dbl dMap_tmp;
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    for (unsigned int j=0;j<i;j++)
      {
	dMap_tmp[i][j] = 0.5*(dMap[i][j]+dMap[j][i]);
	dMap_tmp[j][i]=dMap_tmp[i][j];
      }
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    dMap_tmp[i][i]=0.0;
  
  double Err=0.0;
  while (check_for_zero(&dMap_tmp, &Err))
    embeddingError.push_back(Err);

  eDIM=embeddingError.size(); 
}


//--------------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   unsigned int RPT)
{
  TOLERANCE=0.000001;
  num_elements=S.size();

  /** 
      Create distance matrix dMap 
      via semantic annihilation
  */
#pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int i=0;i<num_elements;i++)
    for (unsigned int j=0;j<num_elements;j++)
      {
	double SUM=0.0;
	for (unsigned int r=0; r < RPT;r++)
	  {
	    Symbolic_string_ tmp(S[i] + !S[j]);
	    tmp.get_norm();
	    SUM+=tmp.norm;
	  }
	dMap[i][j] = SUM/(RPT+0.0);
      }

  /**
     Create DMap_tmp as the 
     zero-diagonal symmetricized version 
     of the distance matrix dMap;
  */
  matrix_dbl dMap_tmp;
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    for (unsigned int j=0;j<i;j++)
      {
	dMap_tmp[i][j] = 0.5*(dMap[i][j]+dMap[j][i]);
	dMap_tmp[j][i]=dMap_tmp[i][j];
      }
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    dMap_tmp[i][i]=0.0;
  
  double Err=0.0;
  while (check_for_zero(&dMap_tmp, &Err))
    embeddingError.push_back(Err);

  eDIM=embeddingError.size(); 
}
//--------------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   unsigned int RPT,
					   unsigned int NORM_DEPTH)
{
  TOLERANCE=0.000001;
  num_elements=S.size();

  /** 
      Create distance matrix dMap 
      via semantic annihilation
  */
  vector<double> PLC_(num_elements*num_elements,0.0);
  
#pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int i=0;i<num_elements;i++)
    for (unsigned int j=0;j<num_elements;j++)
      {
	double SUM=0.0;
	for (unsigned int r=0; r < RPT;r++)
	  {
	    Symbolic_string_ tmp(S[i] + !S[j]);
	    tmp.get_reflected_norm(NORM_DEPTH);
	    SUM+=tmp.norm;
	  }
	PLC_[i+j*num_elements] = SUM/(RPT+0.0);
      }

  for (unsigned int i=0;i<num_elements;i++)
    for (unsigned int j=0;j<num_elements;j++)
      dMap[i][j] =PLC_[i+j*num_elements];
  
  /**
     Create DMap_tmp as the 
     zero-diagonal symmetricized version 
     of the distance matrix dMap;
  */
  matrix_dbl dMap_tmp;
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    for (unsigned int j=0;j<i;j++)
      {
	dMap_tmp[i][j] = 0.5*(dMap[i][j]+dMap[j][i]);
	dMap_tmp[j][i]=dMap_tmp[i][j];
      }
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
    dMap_tmp[i][i]=0.0;
  
  double Err=0.0;
  while (check_for_zero(&dMap_tmp, &Err))
    embeddingError.push_back(Err);

  eDIM=embeddingError.size(); 
}
//--------------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   unsigned int RPT,
					   unsigned int NORM_DEPTH,
					   unsigned int iO, 
					   unsigned int iE,
					   unsigned int jO, 
					   unsigned int jE)
{
  TOLERANCE=0.000001;
  num_elements=S.size();



  /** 
      Create distance matrix dMap 
      via semantic annihilation
  */
#pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int i=iO;i<iE;i++)
    for (unsigned int j=jO;j<jE;j++)
      {
	double SUM=0.0;
	for (unsigned int r=0; r < RPT;r++)
	  {
	    Symbolic_string_ tmp(!S[i] + S[j]);
	    tmp.get_reflected_norm(NORM_DEPTH);
	    SUM+=tmp.norm;
	  }
	dMap[i][j] = SUM/(RPT+0.0);
      }

}

//--------------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   unsigned int RPT,
					   unsigned int NORM_DEPTH,
					   bool SAE_
					   )
{
  TOLERANCE=0.000001;
  num_elements=S.size();

  if (!SAE_)
    {

      /** 
	  Create distance matrix dMap 
	  via semantic annihilation
      */
      #pragma omp parallel for schedule(dynamic,1) collapse(2)
      for (unsigned int i=0;i<num_elements;i++)
	for (unsigned int j=0;j<num_elements;j++)
	  {
	    double SUM=0.0;
	    for (unsigned int r=0; r < RPT;r++)
	      {
		Symbolic_string_ tmp(S[i] + !S[j]);
		tmp.get_reflected_norm(NORM_DEPTH);
		SUM+=tmp.norm;
	      }
	    dMap[i][j] = SUM/(RPT+0.0);
	  }

      /**
	 Create DMap_tmp as the 
	 zero-diagonal symmetricized version 
	 of the distance matrix dMap;
      */
      matrix_dbl dMap_tmp;
      for (unsigned int i=0;i<(unsigned int)num_elements;i++)
	for (unsigned int j=0;j<i;j++)
	  {
	    dMap_tmp[i][j] = 0.5*(dMap[i][j]+dMap[j][i]);
	    dMap_tmp[j][i]=dMap_tmp[i][j];
	  }
      for (unsigned int i=0;i<(unsigned int)num_elements;i++)
	dMap_tmp[i][i]=0.0;
  
      double Err=0.0;
      while (check_for_zero(&dMap_tmp, &Err))
	embeddingError.push_back(Err);

      eDIM=embeddingError.size(); 
    }
  else
    {
      for (unsigned int j=0;j<num_elements;j++)
	{
	  double SUM=0.0;
	    #pragma omp parallel for
	    for (unsigned int r=0; r < RPT;r++)
	      {
		Symbolic_string_ tmp(S[j] + !S[j]);
		tmp.get_reflected_norm(NORM_DEPTH);
		SUM+=tmp.norm;
	      }
	    sae.push_back(SUM/(RPT+0.0));
	}

    }
}

//---------------------------------------------------------------------

Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &SI,
					   vector <Symbolic_string_> &SJ,
					   unsigned int RPT,
					   unsigned int NORM_DEPTH)
{
  TOLERANCE=0.000001;
  unsigned int num_elementsI=SI.size();
  unsigned int num_elementsJ=SJ.size();
  num_elements = num_elementsI;
  /** 
      Create distance matrix dMap 
      via semantic annihilation
  */

  vector<double> PLC_(num_elementsI*num_elementsJ,0.0);

#pragma omp parallel for schedule(dynamic,1) collapse(2) 
  for (unsigned int i=0;i<num_elementsI;i++)
    for (unsigned int j=0;j<num_elementsJ;j++)
      {
	double SUM=0.0;
	for (unsigned int r=0; r < RPT;r++)
	  {
	    Symbolic_string_ tmp(SI[i] + !SJ[j]);
	    tmp.get_reflected_norm(NORM_DEPTH);
	    SUM+=tmp.norm;
	  }
	PLC_[i+j*num_elementsI] = SUM/(RPT+0.0);
      }
    
  for (unsigned int i=0;i<num_elementsI;i++)
    for (unsigned int j=0;j<num_elementsJ;j++)
      dMap[i][j] =PLC_[i+j*num_elementsI];
  
}

//--------------------------------------------------------------------------



Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   unsigned int RPT,
					   unsigned int NORM_DEPTH,
					   bool ONLY_PAST,
					   int H_SPEC )
{
  if (ONLY_PAST)
    {
      TOLERANCE=0.000001;
      num_elements=S.size();
      for (unsigned int i=1;i<num_elements;i++)
	hdiff.push_back(1.0);

#pragma omp parallel for
      for (unsigned int i=H_SPEC;i<num_elements;i++)
	{
	  for (unsigned int j=i-H_SPEC;j<i;j++)
	    {
	      double SUM=0.0;
	      for (unsigned int r=0; r < RPT;r++)
		{
		  Symbolic_string_ tmp(S[i] + !S[j]);
		  tmp.get_reflected_norm(NORM_DEPTH);
		  SUM+=tmp.norm;
		}
	      hdiff[i] *= SUM/(RPT+0.0);
	    }
	  // hdiff[i] /= H_SPEC;
	}
    }  
}
 
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
/*
Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S, 
					   vector <Symbolic_string_> &S_LIB, 
					   unsigned int RPT)
{
  TOLERANCE=0.000001;
  num_elements=S.size();
  unsigned int num_elements_LIB=S_LIB.size();


    #pragma omp parallel for
    for (unsigned int j=0;j<num_elements_LIB;j++)
      for (unsigned int i=0;i<num_elements;i++)
	{
	  double SUM=0.0;
	  for (unsigned int r=0; r < RPT;r++)
	    {
	      Symbolic_string_ tmp(S[i] + !S_LIB[j]);
	      tmp.get_norm();
	      SUM+=tmp.norm;
	    }
	  dMap[i][j] = SUM/(RPT+0.0);
	}
}
*/
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
/*
  void getDistance(unsigned int index_i,
  unsigned int index_j,
  map <unsigned int, map <unsigned int, double> > *H,
  unsigned int RPT,
  vector <Symbolic_string_> *S,
  unsigned int tBLOCK,
  unsigned int Hsize )
  {
  unsigned int imax,jmax;
  if ((index_i*tBLOCK + tBLOCK) > Hsize)
  imax = Hsize;
  else
  imax = index_i*tBLOCK + tBLOCK;
  if ((index_j*tBLOCK + tBLOCK) > Hsize)
  jmax = Hsize;
  else
  jmax = index_j*tBLOCK + tBLOCK;

  vector <double> value_tmp;
  unsigned int count=0;

  for (unsigned int ind_i=index_i*tBLOCK;ind_i<imax;ind_i++)
  for (unsigned int ind_j=index_j*tBLOCK;ind_j<jmax;ind_j++)
  {
  double SUM=0.0;
  for (unsigned int r=0; r < RPT;r++)
  {
  Symbolic_string_ tmp_i((*S)[ind_i]);
  Symbolic_string_ tmp_j((*S)[ind_j]);

  Symbolic_string_ tmp(tmp_i + ~tmp_j);
  //Symbolic_string_ tmp((*S)[ind_i] + !(*S)[ind_j]);
  tmp.get_norm();
  SUM+=tmp.norm;
  }
  value_tmp.push_back(SUM/(RPT+0.0));
  }

  count=0;
  for (unsigned int ind_i=index_i*tBLOCK;ind_i<imax;ind_i++)
  for (unsigned int ind_j=index_j*tBLOCK;ind_j<jmax;ind_j++)
  (*H)[ind_i][ind_j] = value_tmp[count++];

  }

  //--------------------------------------------------
  void exec_th(map < unsigned int, unsigned int > map_curr,
  map <unsigned int, map <unsigned int, double> > *H,
  unsigned int RPT,
  vector <Symbolic_string_> *S,
  unsigned int tBLOCK)
  {
  thread_group Threads;
  for(map < unsigned int, unsigned int >::iterator itr=map_curr.begin();
  itr!=map_curr.end();
  ++itr)
  {
  Threads.create_thread(bind(&getDistance,
  itr->first,
  itr->second,
  H,RPT,S,tBLOCK,S->size()));
  // cout << Threads.size() << endl;
  usleep(1000);
  }
  Threads.join_all();
  };
*/
//--------------------------------------------------
/**
   Implements algorithm for choosing index pairs {(i,j), (i2,j2),.. (i3,j3)} from
   a matrix, such that no index is repeated i.e. if (1,3) is one of the indiex pairs
   then the remaining pairs cannot have either a "1" or a "3"
   as either of the two index entries.
   This is important, since now we do not need to use mutex locks when
   accessing resource maps indexed by (i,j) in multi-threaded calls, 
   since no entry is repeated. 
**/
/*void getIndices(unsigned int N_t, 
  map < pair<unsigned int,unsigned int>, bool > &BB, 
  map < unsigned int, unsigned int > &map_curr)
  {
  set < unsigned int > indices_curr;
  unsigned int num_th=0;
  while (num_th < N_t)
  {
  map < pair<unsigned int,unsigned int>, 
  bool >::iterator map_itr=BB.begin();
  bool index_Found=false;
  while (map_itr != BB.end())
  {	  
  pair <unsigned int, unsigned int> ind_ij=map_itr->first;
  if ((indices_curr.find(ind_ij.first)
  ==indices_curr.end()) 
  &&  (indices_curr.find(ind_ij.second)
  ==indices_curr.end()))
  {
  indices_curr.insert(ind_ij.first);
  indices_curr.insert(ind_ij.second);
	      
  map_curr[ind_ij.first]=ind_ij.second;
  BB.erase(map_itr);
  index_Found=true;
  break;
  }
  ++map_itr;
  }
  if (!index_Found)
  break;
  else
  num_th++;
  }
  };

  //--------------------------------------------------------------------------

  Set_symbolic_string_::Set_symbolic_string_(vector <Symbolic_string_> &S,
  unsigned int RPT,
  unsigned int Num_thrds,
  unsigned int tBLOCK)
  {
  TOLERANCE=0.000001;
  num_elements=S.size();

  for (unsigned int i=0;i < S.size();i++)
  for (unsigned int j=0; j< S.size();j++)
  dMap[i][j]=0.0;

  unsigned int N_d = ceil((S.size()+0.0/tBLOCK+0.0));
  map < pair<unsigned int,unsigned int>, bool > BB;
  for (unsigned int i=0; i < N_d;i++)
  for (unsigned int j=0;j < N_d;j++)
  BB[make_pair(i,j)]=true;

  while (!BB.empty())
  {
  map < unsigned int, unsigned int > map_curr;
  getIndices(Num_thrds,BB,map_curr);
  exec_th(map_curr,&dMap,RPT,&S,tBLOCK);
  }

  matrix_dbl dMap_tmp;
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
  for (unsigned int j=0;j<i;j++)
  {
  dMap_tmp[i][j] = 0.5*(dMap[i][j]+dMap[j][i]);
  dMap_tmp[j][i]=dMap_tmp[i][j];
  }
  for (unsigned int i=0;i<(unsigned int)num_elements;i++)
  dMap_tmp[i][i]=0.0;
  
  double Err=0.0;
  while (check_for_zero(&dMap_tmp, &Err))
  embeddingError.push_back(Err);

  eDIM=embeddingError.size(); 
  }
*/
/** ------------------------------- */


/** ------------------------------- */

bool Set_symbolic_string_::check_for_zero(matrix_dbl *mdata, double *Err)
{
  pair <unsigned int, unsigned int> Generator(0,0);
  get_generator(mdata, &Generator, Err);

  if (*Err <= TOLERANCE)
    return false;

  map <unsigned int, double> crd_tmp;
  unsigned int i=Generator.first;
  unsigned int k=Generator.second;

  for (unsigned int j=0; j < (unsigned int)num_elements; j++)
    crd_tmp[j]=((*mdata)[i][j]*(*mdata)[i][j] 
		+ (*mdata)[i][k]*(*mdata)[i][k] 
		- (*mdata)[j][k]*(*mdata)[j][k]) / (2*(*mdata)[i][k]);
  embeddingCoordinates[embeddingCoordinates.size()]=crd_tmp;

#pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int r=0; r < (unsigned int)num_elements; r++)
    for (unsigned int c=0; c < (unsigned int)num_elements; c++)
      (*mdata)[r][c] =  sqrt(abs((*mdata)[r][c]*(*mdata)[r][c] 
				 - (crd_tmp[c]-crd_tmp[r])
				 *(crd_tmp[c]-crd_tmp[r])));
  return true;
}

/** ------------------------------- */

void Set_symbolic_string_::get_generator(matrix_dbl *mat,
					 pair <unsigned int, 
					 unsigned int> *gen,
					 double *err)
{
  *err=0.0;

  #pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int i=0; i < (unsigned int)num_elements; i++)
    for (unsigned int j=0; j<(unsigned int)num_elements; j++)
      if (*err<(*mat)[i][j])
	if (j>i)
	  {
	    *err=(*mat)[i][j];
	    gen->first=i;
	    gen->second=j;
	  }
}
//---------------------------------------------------------------------

matrix_dbl Set_symbolic_string_::embedding_coordinates(const unsigned int I)
{
  unsigned int dim=I;
  if (dim > eDIM)
    dim=eDIM;
  /*
    if (dim==eDIM)
    return embeddingCoordinates;
    else
    {
  */
  matrix_dbl cmap;
  for (unsigned int i=0; i<dim; i++)
    for (unsigned int j=0; j<num_elements; j++)
      cmap[j][i]=embeddingCoordinates[i][j];
  return cmap;
  // }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

matrix_dbl Sippl_embed::embedding_coordinates(const unsigned int I)
{
  unsigned int dim=I;
  if (dim > eDIM)
    dim=eDIM;
  
  if (dim==eDIM)
    return embeddingCoordinates;
  else
    {
  
      matrix_dbl cmap;
      for (unsigned int i=0; i<dim; i++)
	for (unsigned int j=0; j<embeddingCoordinates[0].size(); j++)
	  cmap[i][j]=embeddingCoordinates[i][j];
      return cmap;
    }
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------


Sippl_embed::Sippl_embed(matrix_dbl *H)
{
  TOLERANCE=0.00000001;
  matrix_dbl symmetric_Data;
  for (unsigned int i=0;i<(unsigned int)H->size();i++)
    for (unsigned int j=0;j<i;j++)
      {
	symmetric_Data[i][j] = 0.5*((*H)[i][j]+(*H)[j][i]);
	symmetric_Data[j][i] = symmetric_Data[i][j];
      }
  for (unsigned int i=0;i<(unsigned int)H->size();i++)
    symmetric_Data[i][i]=0.0;
  
  double Err=0.0;
  while (check_for_zero(&symmetric_Data, &Err))
    embeddingError.push_back(Err);

  eDIM=embeddingError.size(); 

}


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
bool Sippl_embed::check_for_zero(matrix_dbl *mdata, double *Err)
{
  unsigned int num_elements=mdata->size();
  pair <unsigned int, unsigned int> Generator(0,0);
  get_generator(mdata, &Generator, Err);

  if (*Err <= TOLERANCE)
    return false;

  map <unsigned int, double> crd_tmp;
  unsigned int i=Generator.first;
  unsigned int k=Generator.second;

  for (unsigned int j=0; j < num_elements; j++)
    crd_tmp[j]=((*mdata)[i][j]*(*mdata)[i][j] 
		+ (*mdata)[i][k]*(*mdata)[i][k] 
		- (*mdata)[j][k]*(*mdata)[j][k]) / (2*(*mdata)[i][k]);
  embeddingCoordinates[embeddingCoordinates.size()]=crd_tmp;

  #pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int r=0; r < num_elements; r++)
    for (unsigned int c=0; c < num_elements; c++)
      (*mdata)[r][c] =  sqrt(abs((*mdata)[r][c]*(*mdata)[r][c] 
				 - (crd_tmp[c]-crd_tmp[r])
				 *(crd_tmp[c]-crd_tmp[r])));
  return true;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void Sippl_embed::get_generator(matrix_dbl *mat,
				pair <unsigned int, 
				unsigned int> *gen,
				double *err)
{
  *err=0.0;
  unsigned int num_elements=mat->size();

  #pragma omp parallel for schedule(dynamic,1) collapse(2)
  for (unsigned int i=0; i < num_elements; i++)
    for (unsigned int j=0; j <num_elements; j++)
      if (*err<(*mat)[i][j])
	if (j>i)
	  {
	    *err=(*mat)[i][j];
	    gen->first=i;
	    gen->second=j;
	  }
}



//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
unsigned long int random_seed()
{
  // return ++global_seed;

  unsigned int seed;
  //string RTYPE="strict";
  string RTYPE="clock";

  if (RTYPE=="strict")
    {
      struct timeval tv;
      FILE *devrandom;
      
      if ((devrandom = fopen("/dev/urandom","r")) == NULL) 
	{
	  gettimeofday(&tv,0);
	  seed = tv.tv_sec + tv.tv_usec;
	} 
      else 
	{
	  fread(&seed,sizeof(seed),1,devrandom);
	  fclose(devrandom);
	} 
    }
  else
    {
      struct timeval tv;
      gettimeofday(&tv,0);
      seed = tv.tv_sec + tv.tv_usec;
    }
  return(seed);
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//   genESeSS sources to follow
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------


//------------------------------------
//------------------------------------
vector <double> mk_stochastic(map <symbol, unsigned int > &H)
{
  float S=0.0;
  vector <double> v;

  for(map <symbol, unsigned int >::iterator itr=H.begin();
      itr != H.end();
      ++itr)
    {
      S+=itr->second;
      while (itr->first >= (symbol) v.size())
	v.push_back(0.0);
    }

  if (S > 0.0)
    {
      S=1/S;
      for(map <symbol, unsigned int >::iterator itr=H.begin();
	  itr != H.end();
	  ++itr)
	v[itr->first] = S * itr->second;
    }
  return v;
}
//------------------------------------
//------------------------------------
void mk_stochastic(phi_data_type_ &Phi,stoch_phi_data_type_ &sPhi)
{
  for(phi_data_type_::iterator itr=Phi.begin(); 
      itr!=Phi.end();
      ++itr)
    sPhi[itr->first] = mk_stochastic(itr->second); 
}
//------------------------------------
//------------------------------------
double l2dev(map <symbol, unsigned int > &H1, map <symbol, unsigned int > &H2)
{
  vector <double> v1(mk_stochastic(H1));
  vector <double> v2(mk_stochastic(H2));
  if (v1.size() != v2.size())
    return -1;
  double S=0.0;
  for(unsigned int i=0; i < v1.size();++i)
    S+=(v1[i]-v2[i])*(v1[i]-v2[i]);
  return S;
}
//------------------------------------
//------------------------------------
bool zero(map <symbol, unsigned int > &H)
{
  for(map <symbol, unsigned int >::iterator itr=H.begin();
      itr != H.end();
      ++itr)
    if (itr->second != 0)
      return false;
  return true;
}
//------------------------------------
//------------------------------------

string genESeSS::_color(string str)
{
  if (COL_PRINT)
    {
      int forecolor= _semantic_BLACK;
      int backcolor= _semantic_B_WHITE;
      char buff[1024];
      sprintf(buff,"%c[%d;%d;%dm",27,_semantic_RESET,forecolor,backcolor);
      //sprintf(buff,"%c[%d;%dm",27,_semantic_BOLD,forecolor);
      str=buff+str;
      sprintf(buff,"%c[%dm",27,_semantic_RESET);
      str+=buff;
    }
  return str;
};

//------------------------------------
//------------------------------------
symbol_list_ & genESeSS::inc(symbol_list_ &s)
{
  if (s.empty())
    {
      s.push_back(symbol (0));
      return s;
    }
  symbol sigma=s.back();
  s.pop_back();

  if (sigma++ < alphabet-1)
    s.push_back(sigma);
  else
    {
      s = inc(s);
      s.push_back(symbol (0));
    }
  return s;
}
//------------------------------------
//------------------------------------
void genESeSS_x::getPhi(symbol_list_ omega)
{
  if (!Phi[omega].empty())
    return;
  if (omega.empty())
    {
      /**	 Treat empty string separately      */
      for(unsigned int i=0; 
	  i < data_size;
	  ++i)
	{
	  Phi[omega][data_B[i]]++; 
	  //reading next sym from str B
	  Datapin[omega][data_A[i]].push_back(i); 
	  //pinning or positions must be in str A
	}
      return;
    }

  unsigned int omega_Len=omega.size();
  symbol sym=omega.back();
  symbol_list_ omega0(omega);
  omega0.pop_back();

  /**	 Recursive call to getPhi()      */ 
  if (Phi[omega0].empty())
    getPhi(omega0); 

  for(vector < unsigned int >::iterator itr
	=Datapin[omega0][sym].begin();
      itr!=Datapin[omega0][sym].end();
      ++itr)
    if (*itr+omega_Len < data_size)
      {
	Phi[omega][data_B[*itr+omega_Len]]++;
	Datapin[omega][data_A[*itr+omega_Len]].push_back(*itr);
      }
  for (map <symbol, unsigned int>::iterator itr=Phi[omega0].begin();
       itr!=Phi[omega0].end();
       ++itr)
    {    
      map <symbol, unsigned int>::iterator itr1=Phi[omega].find(itr->first);
      if (itr1 == Phi[omega].end())
	Phi[omega][itr->first]=0;
    }
}
//------------------------------------
//------------------------------------
void genESeSS::getPhi(symbol_list_ omega)
{
  if (!Phi[omega].empty())
    return;
  if (omega.empty())
    {
      /**	 Treat empty string separately      */
      for(unsigned int i=0; 
	  i <data.size();
	  ++i)
	{
	  Phi[omega][data[i]]++;
	  Datapin[omega][data[i]].push_back(i);
	}
      return;
    }

  unsigned int omega_Len=omega.size();
  symbol sym=omega.back();
  symbol_list_ omega0(omega);
  omega0.pop_back();

  /**	 Recursive call to getPhi()      */ 
  if (Phi[omega0].empty())
    getPhi(omega0); 

  for(vector < unsigned int >::iterator itr
	=Datapin[omega0][sym].begin();
      itr!=Datapin[omega0][sym].end();
      ++itr)
    if (*itr+omega_Len < data.size())
      {
	Phi[omega][data[*itr+omega_Len]]++;
	Datapin[omega][data[*itr+omega_Len]].push_back(*itr);
      }
  for (map <symbol, unsigned int>::iterator itr=Phi[omega0].begin();
       itr!=Phi[omega0].end();
       ++itr)
    {    
      map <symbol, unsigned int>::iterator itr1=Phi[omega].find(itr->first);
      if (itr1 == Phi[omega].end())
	Phi[omega][itr->first]=0;
    }
}
//------------------------------------
//------------------------------------
/*symbol_list_ genESeSS::getSync(symbol_list_ lambda_init_)
{
  double loop_count=0;
  symbol_list_ lambda(lambda_init_);
  getPhi(lambda);

  while(1)
    {
      symbol next_sym(0);
      double Min_dev=1000;
      for(symbol sigma(0); sigma < alphabet;++sigma)
	{
	  symbol_list_ omega0(lambda);
	  omega0.push_back(sigma);
	  getPhi(omega0);
	  double dev_this=l2dev(Phi[lambda],Phi[omega0]);
	  if (dev_this < Min_dev)
	    {
	      Min_dev=dev_this;
	      next_sym=sigma;
	    }
	}

      if (!zero(Phi[lambda]))   // zero phi correction
	lambda.push_back(next_sym);
      else
	break;

      if ((Min_dev <= MIN_DEV_SYNC_SEARCH_) 
	  || (loop_count++ > LOOP_COUNT_SYNC_SEARCH_))
	break;
    }
  //cout << lambda << " " << loop_count << " " << LOOP_COUNT_SYNC_SEARCH_  << endl;
  return lambda;
} 
*/
//------------------------------------
//------------------------------------
/*symbol_list_ genESeSS::getSync()
{
  Phi.clear();
  symbol_list_ lambda;
  unsigned int sync_pref_len=0;
  map <symbol_list_, symbol_list_> M_syn;
  while ( sync_pref_len++ < SYN_PREF_LENGTH_)
    {
      symbol_list_ omega(getSync(lambda));
      if (!zero(Phi[omega]))
	M_syn[omega]=lambda;
      inc(lambda);
    }

  if (M_syn.empty())
    return symbol_list_ ();
  else
    return M_syn.begin()->first;
}
*/
//------------------------------------
//------------------------------------
symbol_list_ genESeSS::getSync()
{
  unsigned int OCC_STOP=2;
  Phi.clear();
  bool SYNC=false;
  symbol_list_ lambda;
  unsigned int sync_pref_len=0;

  while ( sync_pref_len++ < SYN_PREF_LENGTH_)
    {
      getPhi(lambda);
      vector <double> v0 = mk_stochastic(Phi[lambda]);
		
      double distance=0.0;
      for(symbol sigma(0); sigma < alphabet;++sigma)
	{
	  symbol_list_ omega_H;
	  omega_H.push_back(sigma);
	  copy(lambda.begin(),lambda.end(),back_inserter(omega_H));
	  getPhi(omega_H);
			
	  vector <double> v1;
	  if (zero(Phi[omega_H]))
	    {
	      if (lambda.size() < OCC_STOP)
		{
		  SYNC=true;
		  break;
		}
	      v1 = v0;
	    }
	  else
	    v1 = mk_stochastic(Phi[omega_H]);
			
	  for(unsigned int i=0; i < alphabet;++i)
	    distance += fabs(v0[i]-v1[i]);		
	}
      distance /= (alphabet*alphabet+0.0);
      if (distance <= MIN_DEV_SYNC_SEARCH_)
	{
	  SYNC=true;
	  break;
	}
      else
	inc(lambda);
    }
  if (SYNC)
    return lambda;
  else
    return symbol_list_ ();
}
//------------------------------------
//------------------------------------
void genESeSS::getAut()
{
  //lock_guard<mutex> guard(mtx_);

  for (double epsilon=EPS_MIN_; epsilon <= EPS_MAX_; epsilon+=EPS_STEP_)
    {
      vector< double> eps_tmp;
      connx autR;
      map < vector <double> , 
	    state, 
	    eps_compare_ > structData(eps_compare_((const double)epsilon));

      structData[mk_stochastic(Phi[omega_syn])] = 0;
      map<state, symbol_list_> stateSignature;
      stateSignature[0]=omega_syn;
      symbol_list_ lambda;


      //cout << Phi << endl;
      // cout << omega_syn <<  " ---- " << mk_stochastic(Phi[omega_syn])    << endl;

      unsigned int num_states=1, curr_state=0;
      while(curr_state < num_states)
	{
	  for(symbol sigma(0); sigma < alphabet;++sigma)
	    {
	      lambda = stateSignature[curr_state];
	      lambda.push_back(sigma);
	      getPhi(lambda);

	      //cout << lambda <<  " ---- " << mk_stochastic(Phi[lambda])    << endl;

	      if (!zero(Phi[lambda]))
		if (structData.find(mk_stochastic(Phi[lambda])) 
		    != structData.end())
		  autR[curr_state][sigma] 
		    = (*(structData.find(mk_stochastic(Phi[lambda])))).second;
		else
		  {
		    autR[curr_state][sigma] 
		      = (++num_states)-1;
		    structData[mk_stochastic(Phi[lambda])] 
		      = num_states-1;
		    stateSignature[num_states-1]=lambda;
		  }
	      else
		autR[curr_state][sigma]=-1;
	    }
	  curr_state++;
	}

      vector < connx > autV(getComponent(autR));
      if (!autV.empty())
	{
	  eps_tmp=vector <double> (autV.size(),epsilon);
	  copy(eps_tmp.begin(), eps_tmp.end(), back_inserter(eps_vec));
	  copy(autV.begin(), autV.end(), back_inserter(aut_vec));
	}
    }
  if (aut_vec.empty())
    NO_STRONG_COMPONENTS_=true;
};
//------------------------------------
//------------------------------------
vector < connx > genESeSS::getComponent(connx Aut)
{
  vector <connx> CAut;
  typedef adjacency_list<vecS, 
			 vecS,
			 bidirectionalS, 
			 no_property,
			 property<edge_weight_t, float> > Graph;
  typedef std::pair<int,int> Edge;

  int states = Aut.size();
  unsigned int num_neg_entries = 0;

  for (int i = 0;i < states; i++)
    for (symbol j(0);j < alphabet;j++)
      if (Aut[i][j] == -1)
	num_neg_entries += 1;

  const int num_vertices = states;
  unsigned int num_edges = num_vertices*alphabet - num_neg_entries;
  int edge_weight[num_edges];
  Edge edge_array[num_edges];
  unsigned int pos = 0;

  for (int i = 0; i < states; i++)
    for (symbol j(0); j <  alphabet; j++)
      if (Aut[i][j] != -1)
	{
	  edge_array[pos] = Edge(i,Aut[i][j]);
	  edge_weight[pos++] = j;
	}
  Graph g(edge_array, edge_array + num_edges, edge_weight, num_vertices);
  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  vector<int> component(num_vertices), discover_time(num_vertices);
  vector<default_color_type> color(num_vertices);
  vector<Vertex> root(num_vertices);


  /*      
	  int num_components = strong_components(g, &component[0], 
	  root_map(&root[0]).
	  color_map(&color[0]).
	  discover_time_map(&discover_time[0]));

	  // boost 1.48 old code
	  */


  int num_components = 
    strong_components(g, 
		      make_iterator_property_map(component.begin(), 
						 get(vertex_index, g)), 
		      root_map(make_iterator_property_map(root.begin(), 
							  get(vertex_index, g))).
		      color_map(make_iterator_property_map(color.begin(), 
							   get(vertex_index, g))).
		      discover_time_map(make_iterator_property_map(discover_time.begin(),
								   get(vertex_index, g))));

  if (num_components == 0)//no strong component present
    return CAut;
  if (num_components == 1)
    {
      CAut.push_back(Aut);
      return CAut;
    }
  
  map <int, vector <state> > componentMap;
  for (int i=0; i < states;++i)
    componentMap[component[i]].push_back(i);
  bool SINGLE_STATE_C=false;

  for ( map <int, vector <state> >::iterator itr=componentMap.begin();
	itr!=componentMap.end();
	++itr)
    {
      connx sub_aut;
      map <state,state> state_rename;
      for (state i=0; (unsigned int) i < itr->second.size();++i)
	state_rename[itr->second[i]]=i;

      for (state i=0;(unsigned int) i<itr->second.size();++i)
	for(symbol sigma(0); sigma <alphabet;++sigma)
	  sub_aut[i][sigma] = state_rename[Aut[itr->second[i]][sigma]];
      if (sub_aut.size() > 1)
	CAut.push_back(sub_aut);
      else
	SINGLE_STATE_C=true;
    }

  if (SINGLE_STATE_C)
    {
      connx single_state_aut;
      for(symbol sigma(0); sigma <alphabet;++sigma)
	single_state_aut[0][sigma]=0;
      CAut.push_back(single_state_aut );
    }
  return CAut;
}
//------------------------------------
//------------------------------------ 
void genESeSS::getPit()
{
  //lock_guard<mutex> guard(mtx_);

  if (!NO_STRONG_COMPONENTS_)
    {
      for(vector <connx>::iterator itr=aut_vec.begin();
	  itr!=aut_vec.end();
	  ++itr)
	{
	  pitilde pit_tmp;
	  map<state,map<symbol, unsigned int> > traversal_count;
	  state curr_state_init=0;

	  while (1)
	    {
	      state curr_state=curr_state_init++;
	      for (symbol_list_::iterator itr1=data.begin();
		   itr1!=data.end();
		   ++itr1)
		if ((*itr)[curr_state][*itr1] >= 0)
		  {
		    traversal_count[curr_state][*itr1]++;
		    curr_state=(*itr)[curr_state][*itr1];
		  }
	      if (traversal_count.size() == itr->size())
		break;
	      if (curr_state_init == (int) itr->size())
		{
		  cout << "EXCEPTION: aut and pit size mismatch" 
		       << " # aut size = " << (int) itr->size()
		       << " Tcount size " << traversal_count.size() 
		       << " sync " << omega_syn 
		       << " " << *itr
		       << endl;
		  break;
		}
	    }

	  /*	  for(map<state,map<symbol, unsigned int> >::iterator itr2
		=traversal_count.begin();
	      itr2!=traversal_count.end();
	      itr2++)
	    {
	      vector <double> pit_row(mk_stochastic(itr2->second));
	      while (pit_row.size() < alphabet)
		pit_row.push_back(0.0);
	      pit_tmp[itr2->first]=pit_row;
	    }
	  */
	  for (state q=0; q < (int) itr->size(); ++q)
	    {
	      map<symbol, unsigned int> pit_row_count;
	      if (traversal_count.find(q) != traversal_count.end())
		pit_row_count = traversal_count[q];
	      else
		{
		  for (symbol sym(0); sym < alphabet; ++sym)
		    if ((*itr)[q][sym] != -1)
		      pit_row_count[sym]=1;
		    else
		      pit_row_count[sym]=0;
		}
	      for (symbol sym(0); sym < alphabet; ++sym)
		if (pit_row_count.find(sym) == pit_row_count.end())
		  pit_row_count[sym] = 0;

	      vector <double> pit_row(mk_stochastic(pit_row_count));
	      pit_tmp[q]=pit_row;
	    }
	  pit_vec.push_back(pit_tmp);
	}
    }
}
//------------------------------------
//------------------------------------ 
void genESeSS_x::getPit()
{
  //lock_guard<mutex> guard(mtx_);

  if (!NO_STRONG_COMPONENTS_)
    {
      for(vector <connx>::iterator itr=aut_vec.begin();
	  itr!=aut_vec.end();
	  ++itr)
	{
	  pitilde pit_tmp;
	  map<state,map<symbol, unsigned int> > traversal_count;// sym in str B
	  state curr_state_init=0;

	  while (1)
	    {
	      unsigned int data_B_index=0;
	      state curr_state=curr_state_init++;
	      for (symbol_list_::iterator itr1=data_A.begin();
		   itr1!=data_A.end();
		   ++itr1)
		{		  
		  if ((*itr)[curr_state][*itr1] >= 0)
		    {
		      //traversal_count[curr_state][*itr1]++;  //sym in str B
		      traversal_count[curr_state][data_B[data_B_index++]]++;  
		      curr_state=(*itr)[curr_state][*itr1];
		    }
		  if (data_B_index == data_B.size())
		    break;
		}
	      if (traversal_count.size() == itr->size())
		break;
	      if (curr_state_init == (int) itr->size())
		{
		  cout << "EXCEPTION: aut and pit size mismatch" << endl;
		  break;
		}
	    }

	  for (state q=0; q < (int) itr->size(); ++q)
	    {
	      map<symbol, unsigned int> pit_row_count;
	      if (traversal_count.find(q) != traversal_count.end())
		pit_row_count = traversal_count[q];
	      else
		{
		  for (symbol sym(0); sym < alphabet_B; ++sym)
		    if ((*itr)[q][sym] != -1)
		      pit_row_count[sym]=1;
		    else
		      pit_row_count[sym]=0;
		}
	      for (symbol sym(0); sym < alphabet_B; ++sym)
		if (pit_row_count.find(sym) == pit_row_count.end())
		  pit_row_count[sym] = 0;

	      vector <double> pit_row(mk_stochastic(pit_row_count));
	      pit_tmp[q]=pit_row;
	    }
	  pit_vec.push_back(pit_tmp);
	}
    }
}
//------------------------------------
//------------------------------------
void genESeSS_multistream::getPhi(symbol_list_ omega)
{
  if (!Phi[omega].empty())
    return;
  if (omega.empty())
    {
      for(unsigned int stream_num=0; 
	  stream_num < data.size();
	  ++stream_num)
	{
	  pin_data_type_ datapin;
	  for(unsigned int i=0;
	      i<data[stream_num].size();
	      ++i)
	    {
	      Phi[omega][data[stream_num][i]]++;
	      datapin[omega][data[stream_num][i]].push_back(i);
	    }
	  Datapin.push_back(datapin);
	}
      return;
    }

  unsigned int omega_Len=omega.size();
  symbol sym=omega.back();
  symbol_list_ omega0(omega);
  omega0.pop_back();

  if (Phi[omega0].empty())
    getPhi(omega0); 

  for(unsigned int stream_num=0;
      stream_num < data.size();
      ++stream_num)
    for(vector < unsigned int >::iterator itr
	  =Datapin[stream_num][omega0][sym].begin();
	itr!=Datapin[stream_num][omega0][sym].end();
	++itr)
      if (*itr+omega_Len < data[stream_num].size())
	{
	  Phi[omega][data[stream_num][*itr+omega_Len]]++;
	  Datapin[stream_num][omega]
	    [data[stream_num][*itr+omega_Len]].push_back(*itr);
	}
  for (map <symbol, unsigned int>::iterator itr=Phi[omega0].begin();
       itr!=Phi[omega0].end();
       ++itr)
    {    
      map <symbol, unsigned int>::iterator itr1=Phi[omega].find(itr->first);
      if (itr1 == Phi[omega].end())
	Phi[omega][itr->first]=0;
    }
}
//------------------------------------
//------------------------------------
void genESeSS_multistream::getPit()
{
  if (!NO_STRONG_COMPONENTS_)
    {
      for(vector <connx>::iterator itr=aut_vec.begin();
	  itr!=aut_vec.end();
	  ++itr)
	{
	  pitilde pit_tmp;
	  map<state,map<symbol, unsigned int> > traversal_count;
	  state curr_state_init=0;

	  while (1)
	    {
	      state curr_state=curr_state_init++;
	      for(unsigned int stream_num=0;
		  stream_num < data.size();
		  ++stream_num)
		{

		  /*
		    We are keeping the curr_state 
		    when we are reading the next 
		    stream. This is not quite correct.
		    However for small number of data streams
		    which are short, randomizing the 
		    state here causes the pitilde to fluctuate
		    even if the aut remains constant.

		    A better approach would be to generate 
		    a number of models (same aut, but different pit),
		    and the use llk to determine which model
		    fits most of the data streams.-- ixc2018

		   */
		  
		  for (symbol_list_::iterator itr1=data[stream_num].begin();
		       itr1!=data[stream_num].end();
		       ++itr1)
		    if ((*itr)[curr_state][*itr1] >= 0)
		      {
			traversal_count[curr_state][*itr1]++;
			curr_state=(*itr)[curr_state][*itr1];		      
		      }
		}
	      if (traversal_count.size() == itr->size())
		break;
	      if (curr_state_init == (int) itr->size())
		{
		  cout << "EXCEPTION: aut and pit size mismatch" 
		       << " # aut size = " << (int) itr->size()
		       << " Tcount size " << traversal_count.size() 
		       << " sync " << omega_syn 
		       << " " << *itr
		       << endl;
		  break;
		}	      
	    }

	  for (state q=0; q < (int) itr->size(); ++q)
	    {
	      map<symbol, unsigned int> pit_row_count;
	      if (traversal_count.find(q) != traversal_count.end())
		pit_row_count = traversal_count[q];
	      else
		{
		  for (symbol sym(0); sym < alphabet; ++sym)
		    if ((*itr)[q][sym] != -1)
		      pit_row_count[sym]=1;
		    else
		      pit_row_count[sym]=0;
		}

	      for (symbol sym(0); sym < alphabet; ++sym)
		if (pit_row_count.find(sym) == pit_row_count.end())
		  pit_row_count[sym] = 0;

	      vector <double> pit_row(mk_stochastic(pit_row_count));
	      pit_tmp[q]=pit_row;
	    }
	  pit_vec.push_back(pit_tmp);
	}
    }
}
//------------------------------------
//------------------------------------
genESeSS::genESeSS(symbol_list_& dat, unsigned int alph)
{ 
  NO_STRONG_COMPONENTS_=false;

  MIN_DEV_SYNC_SEARCH_=0.001;
  LOOP_COUNT_SYNC_SEARCH_=5;
  SYN_PREF_LENGTH_ = 4;
  EPS_MIN_=0.02;
  EPS_MAX_=0.2;
  EPS_STEP_=0.02;
  ANN_TEST_LEN_=10000;

  alphabet=alph;
  data=dat;
  omega_syn = getSync(); 

  /*
    stoch_phi_data_type_ sPhi2;
    mk_stochastic(Phi,sPhi2);
    cout << sPhi2 << endl;
  */ 
  getAut();
  getPit(); 
  annihilation_error=10000;

  Symbolic_string_ S0(dat);
  
  if (!NO_STRONG_COMPONENTS_)
    for (unsigned int i=0; i < aut_vec.size(); ++i)
      {
	//	cout << i << " " << endl << endl;
	/*	for(unsigned int j=0; j < pit_vec[i].size(); ++j)
		{
		for(unsigned int k=0; k < pit_vec[i][j].size(); ++k)
		cout << pit_vec[i][j][k] << " ";
		cout << endl;
		}
	*/
	//	cout << endl << endl;
	pitilde pit(pit_vec[i]);
	connx aut(aut_vec[i]);

	if (aut.size() > 1)
	  {
	    PFSA G_tmp(pit, aut);
	    Symbolic_string_ Sgen = G_tmp.gen_data(ANN_TEST_LEN_);
	    Symbolic_string_ Ss(S0 + !Sgen);
	    Ss.get_norm(EVALUATION_DEPTH);
	      
	      

	    if (Ss.norm < annihilation_error)
	      {
		G_mc_ = G_tmp;
		annihilation_error = Ss.norm;
		eps=eps_vec[i];
	      }
	  }
	//	else
	// cout << "AUT SS" << endl;

      }
};
//------------------------------------
//------------------------------------
genESeSS::genESeSS(Symbolic_string_ dat, gParam parameters)
{ 
  ANN_TEST_LEN_=parameters.ANN_TEST_LEN_;
  MIN_DEV_SYNC_SEARCH_=parameters.MIN_DEV_SYNC_SEARCH_;
  LOOP_COUNT_SYNC_SEARCH_=parameters.LOOP_COUNT_SYNC_SEARCH_;
  SYN_PREF_LENGTH_ = parameters.SYN_PREF_LENGTH_;
  EPS_MIN_=parameters.EPS_MIN_;
  EPS_MAX_=parameters.EPS_MAX_;
  EPS_STEP_=parameters.EPS_STEP_;
  COL_PRINT=parameters.COL_PRINT;
  EVALUATION_DEPTH=parameters.EVALUATION_DEPTH;
  int NUM_EACH_this=parameters.NUM_EACH;
  int  MIN_STATE_SIZE =parameters.MIN_STATE_SIZE;
  alphabet=dat.get_alphabet();
  data=dat.get_symbol_list();
  omega_syn = getSync(); 

  NO_STRONG_COMPONENTS_=false;
  annihilation_error=10000;

  //double ANN_ERR_MAX=0.3;
  // unsigned int MAX_SYNC_PERTURBS=5;
  // unsigned int sync_perturb_count=0;

  //while (sync_perturb_count++ < MAX_SYNC_PERTURBS)
  // {
      getAut();
      getPit(); 
  
      if (!NO_STRONG_COMPONENTS_)
	{
	  //if only less than MIN_STATE_SIZE is present prevent segfault
	  int max_state_size=0;
	  for (unsigned int i=0; i < aut_vec.size(); ++i)
	    if(aut_vec[i].size() > (unsigned int) max_state_size)
	      max_state_size = aut_vec[i].size();

	  if (max_state_size < MIN_STATE_SIZE)
	    MIN_STATE_SIZE=max_state_size;

	  for (unsigned int i=0; i < aut_vec.size(); ++i)
	    {
	      pitilde pit(pit_vec[i]);
	      connx aut(aut_vec[i]);

	      bool PIT_TRIVIAL=false;
	      double sum_diag=0.0;
	      unsigned int diag_element=0;
	      for  (pitilde::iterator itr=pit.begin();
		    itr!=pit.end();++itr)
		sum_diag += ((itr->second))[diag_element++];
	      if (fabs(sum_diag - pit.size()) < 0.00001)
		PIT_TRIVIAL=true;

	      //int NUM_EACH_this=10;
	      if (!(((int)aut.size() < MIN_STATE_SIZE) || (PIT_TRIVIAL)))
		{
		  PFSA G_tmp(pit, aut);
		  double NORM=0.0;
		  bool LEGACY=false;
		  
		  for (int numeach=0; numeach<NUM_EACH_this;++numeach)
		    {
		      Symbolic_string_ Sgen = G_tmp.gen_data(ANN_TEST_LEN_);
		      
		      if (LEGACY)
			{
			  Symbolic_string_ Ss(dat + !Sgen);
			  Ss.get_norm(EVALUATION_DEPTH);
			  NORM+=Ss.norm;
			}
		      else
			{
			  symbol_list_ omega_i=omega_syn;
			  for(unsigned int w=0;w<=EVALUATION_DEPTH;++w)
			    {
			      Sgen.gen_Phi(omega_i);
			      omega_i=inc(omega_i);
			    }
			  phi_data_type_ Phi_I=Sgen.get_Phi();
			  for(phi_data_type_::iterator itr__=Phi_I.begin();
			      itr__!=Phi_I.end();
			      ++itr__)
			    NORM+=vec_distance(mk_stochastic(itr__->second),
					       mk_stochastic(Phi[itr__->first]));
			}
		    } 
		  if (NORM/(NUM_EACH_this+0.0) < annihilation_error)
		    {
		      G_mc_ = G_tmp;
		      annihilation_error = NORM/(NUM_EACH_this+0.0);
		      eps=eps_vec[i];
		    }
		}
	    }
	}
};
//------------------------------------
//------------------------------------
genESeSS_multistream::genESeSS_multistream(vector< symbol_list_> & dat, 
					   unsigned int alph)
{ 
  NO_STRONG_COMPONENTS_=false;

  MIN_DEV_SYNC_SEARCH_=0.001;
  LOOP_COUNT_SYNC_SEARCH_=5;
  SYN_PREF_LENGTH_ = 4;
  EPS_MIN_=0.02;
  EPS_MAX_=0.2;
  EPS_STEP_=0.02;
  ANN_TEST_LEN_=10000;

  alphabet=alph;
  data=dat;

  omega_syn = getSync(); 
  getAut();
  getPit();

  annihilation_error=10000;

  Symbolic_string_ S0(dat[0]);
  
  if (!NO_STRONG_COMPONENTS_)
    for (unsigned int i=0; i < aut_vec.size(); ++i)
      {
	pitilde pit(pit_vec[i]);
	connx aut(aut_vec[i]);

	if (aut.size() > 0)
	  {
	    PFSA G_tmp(pit, aut);
	    Symbolic_string_ Sgen = G_tmp.gen_data(ANN_TEST_LEN_);
	    Symbolic_string_ Ss(S0 + !Sgen);
	    Ss.get_norm();
	      
	    if (Ss.norm < annihilation_error)
	      {
		G_mc_ = G_tmp;
		annihilation_error = Ss.norm;
		eps=eps_vec[i];
	      }
	  }
	
      }
  

};
//------------------------------------
genESeSS_multistream::genESeSS_multistream(vector< Symbolic_string_> & dat, 
					   gParam parameters)
{ 
  cout << "multi-stream genESeSS.." << endl;
  ANN_TEST_LEN_=parameters.ANN_TEST_LEN_;
  MIN_DEV_SYNC_SEARCH_=parameters.MIN_DEV_SYNC_SEARCH_;
  LOOP_COUNT_SYNC_SEARCH_=parameters.LOOP_COUNT_SYNC_SEARCH_;
  SYN_PREF_LENGTH_ = parameters.SYN_PREF_LENGTH_;
  EPS_MIN_=parameters.EPS_MIN_;
  EPS_MAX_=parameters.EPS_MAX_;
  EPS_STEP_=parameters.EPS_STEP_;
  COL_PRINT=parameters.COL_PRINT;
  NO_STRONG_COMPONENTS_=false; 
  annihilation_error=10000;

  alphabet=0;
  for (unsigned int numstream=0; numstream < dat.size();++numstream)
    if (dat[numstream].get_alphabet() > alphabet)
      alphabet = dat[numstream].get_alphabet();

  EVALUATION_DEPTH=parameters.EVALUATION_DEPTH;
  int NUM_EACH_this=parameters.NUM_EACH;
  int  MIN_STATE_SIZE =parameters.MIN_STATE_SIZE;

  for(unsigned int i=0; i < dat.size();++i)
    data.push_back(dat[i].get_symbol_list());

  omega_syn = getSync(); 

  getAut();
  getPit();

  if (!NO_STRONG_COMPONENTS_)
    {
      //if only less than MIN_STATE_SIZE is present prevent segfault
      int max_state_size=0;
      for (unsigned int i=0; i < aut_vec.size(); ++i)
	if(aut_vec[i].size() > (unsigned int) max_state_size)
	  max_state_size = aut_vec[i].size();

      if (max_state_size < MIN_STATE_SIZE)
	MIN_STATE_SIZE=max_state_size;

      for (unsigned int i=0; i < aut_vec.size(); ++i)
	{
	  pitilde pit(pit_vec[i]);
	  connx aut(aut_vec[i]);

	  bool PIT_TRIVIAL=false;
	  double sum_diag=0.0;
	  unsigned int diag_element=0;
	  for  (pitilde::iterator itr=pit.begin();
		itr!=pit.end();++itr)
	    sum_diag += ((itr->second))[diag_element++];
	  if (fabs(sum_diag - pit.size()) < 0.00001)
	    PIT_TRIVIAL=true;

	  if (!(((int)aut.size() < MIN_STATE_SIZE) || (PIT_TRIVIAL)))
	    {
	      PFSA G_tmp(pit, aut);
	      double NORM=0;

	      for (int numeach=0; numeach<NUM_EACH_this;++numeach)
		{
		  Symbolic_string_ Sgen = G_tmp.gen_data(ANN_TEST_LEN_);

		  symbol_list_ omega_i=omega_syn;
		  for(unsigned int w=0;w<=EVALUATION_DEPTH;++w)
		    {
		      Sgen.gen_Phi(omega_i);
		      omega_i=inc(omega_i);
		    }
		  phi_data_type_ Phi_I=Sgen.get_Phi();
		  double thisNORM=0.0;
		  for(phi_data_type_::iterator itr__=Phi_I.begin();
		      itr__!=Phi_I.end();
		      ++itr__)
		    thisNORM+=vec_distance(mk_stochastic(itr__->second),
				       mk_stochastic(Phi[itr__->first]));

		  thisNORM/= (Phi_I.size()+0.0);
		  NORM+=thisNORM;
		}

	      if (NORM/(NUM_EACH_this+0.0) < annihilation_error)
		{
		  G_mc_ = G_tmp;
		  annihilation_error = NORM/(NUM_EACH_this+0.0);
		  eps=eps_vec[i];
		}
	    }	
	}
    }  
};
//------------------------------------
//------------------------------------
genESeSS_multistream::genESeSS_multistream(vector< symbol_list_> & dat, 
					   gParam parameters)
{ 
  ANN_TEST_LEN_=parameters.ANN_TEST_LEN_;
  MIN_DEV_SYNC_SEARCH_=parameters.MIN_DEV_SYNC_SEARCH_;
  LOOP_COUNT_SYNC_SEARCH_=parameters.LOOP_COUNT_SYNC_SEARCH_;
  SYN_PREF_LENGTH_ = parameters.SYN_PREF_LENGTH_;
  EPS_MIN_=parameters.EPS_MIN_;
  EPS_MAX_=parameters.EPS_MAX_;
  EPS_STEP_=parameters.EPS_STEP_;
  COL_PRINT=parameters.COL_PRINT;
  NO_STRONG_COMPONENTS_=false; 
  EVALUATION_DEPTH=parameters.EVALUATION_DEPTH;

  alphabet=0;
  unsigned int alph=0;
  for (unsigned int numstream=0; numstream < dat.size();++numstream)
    {
      set <symbol> data_set(dat[numstream].begin(),dat[numstream].end());
      if ( (alph=data_set.size()) > alphabet)
	alphabet = alph;
    }
  data=dat;

  omega_syn = getSync(); 
  getAut();
  getPit();

  annihilation_error=10000;

  if (!NO_STRONG_COMPONENTS_)
    for (unsigned int i=0; i < aut_vec.size(); ++i)
      {
	pitilde pit(pit_vec[i]);
	connx aut(aut_vec[i]);
	double NORM=0;

	if (aut.size() > 0)
	  {
	    PFSA G_tmp(pit, aut);
	    Symbolic_string_ Sgen = G_tmp.gen_data(ANN_TEST_LEN_);

	    if (parameters.EXTREMELY_SHORT_)
	      {
		symbol_list_ Smerge;
		for(unsigned int numstream=0; numstream < dat.size();++numstream)
		  copy (dat[numstream].begin(),
			dat[numstream].end(),
			back_inserter(Smerge));
		Symbolic_string_ Ss(Symbolic_string_ (Smerge) + !Sgen);
		Ss.get_norm(EVALUATION_DEPTH);
		NORM+=Ss.norm;
	      }
	    else
	      {
		for(unsigned int numstream=0; numstream < dat.size();++numstream)
		  {
		    Symbolic_string_ Ss(Symbolic_string_ (dat[numstream]) + !Sgen);
		    if (!Ss.get_symbol_list().empty())
		      {
			Ss.get_norm(EVALUATION_DEPTH);
			NORM+=Ss.norm;
		      }
		  }	      	      
		NORM /= (dat.size()+0.0);
	      }
	    if (NORM < annihilation_error)
	      {
		G_mc_ = G_tmp;
		annihilation_error = NORM;
		eps=eps_vec[i];
	      }
	  }	
      }  
};
//------------------------------------
//------------------------------------
genESeSS_x::genESeSS_x(Symbolic_string_ dat1,
		       Symbolic_string_ dat2, 
		       gParam parameters)
{ 
  ANN_TEST_LEN_=parameters.ANN_TEST_LEN_;
  MIN_DEV_SYNC_SEARCH_=parameters.MIN_DEV_SYNC_SEARCH_;
  LOOP_COUNT_SYNC_SEARCH_=parameters.LOOP_COUNT_SYNC_SEARCH_;
  SYN_PREF_LENGTH_ = parameters.SYN_PREF_LENGTH_;
  EPS_MIN_=parameters.EPS_MIN_;
  EPS_MAX_=parameters.EPS_MAX_;
  EPS_STEP_=parameters.EPS_STEP_;
  COL_PRINT=parameters.COL_PRINT;
  USE_GAMMA=parameters.USE_GAMMA;
  EVALUATION_DEPTH=parameters.EVALUATION_DEPTH;
  combined_error=1000000;  
  annihilation_error=1000000;
  local_error=1000000;  
  int NUM_EACH_this=parameters.NUM_EACH;
  NO_STRONG_COMPONENTS_=false;

  alphabet=0;
  alphabet_A=dat1.get_alphabet();
  alphabet_B=dat2.get_alphabet();

  data_A=dat1.get_symbol_list();
  data_B=dat2.get_symbol_list();
  data_size = min(data_A.size(),data_B.size());

  alphabet = alphabet_A;
  omega_syn = getSync();
  getAut();
  getPit(); 

  srand ( time(NULL) );
  Xgamma=0;

  if (!NO_STRONG_COMPONENTS_)
    {
      for (unsigned int i=0; i < aut_vec.size(); ++i)
	{
	  pitilde pit(pit_vec[i]);
	  connx aut(aut_vec[i]);


	  //recalculate pit based on inferred aut structure







	  if (aut.size() > 0)
	    {
	      if (USE_GAMMA)
		{
		  /* Compute X-smashing error */
		  double this_error=0.0;
		  PFSA G;
		  G.set_aut(aut);
		  G.set_pit(pit);               
		  //   #pragma omp parallel for
		  for (int numeach=0; numeach<NUM_EACH_this;++numeach)
		    {
		      Symbolic_string_ Ss( ~dat2 + !(dat1*G));
			  
			  // Yi: modify if we want an SAE version
		      Ss.get_reflected_norm(EVALUATION_DEPTH);
		      this_error+=Ss.norm;
		    }

		  this_error /= (NUM_EACH_this+0.0); /*Measure of global predictability*/
		  double N = this_error;

		  double this_gamma=0;
		  //vector <double> phi_lambda(alphabet_B,0.0);
		  map<symbol, unsigned int> phi_lambda_count;
		  for(unsigned int ib=0; ib<alphabet_B;++ib)
		    phi_lambda_count[symbol (ib)]=0;

		  for(unsigned int ii=0; ii <data_B.size(); ++ii)
		    phi_lambda_count[data_B[ii]]++;

		  vector <double> phi_lambda(mk_stochastic(phi_lambda_count));


		  /*		  
		    for(unsigned int ii=0; ii <alphabet_B; ++ii)
		    phi_lambda[ii] /= (data_B.size()+0.0);
		  */

		  int curr_state=0;
		  vector <double> state_visit(aut.size(),0.0);
		  unsigned int len=data_A.size();
		  for (int numeach=0; numeach<NUM_EACH_this;++numeach)
		    {
		      if (aut.size() > 1)
			curr_state=rand()%(aut.size()-1);
		      for(size_t ii = 0; ii < len; ii++)
			{
			  state_visit[curr_state]++;
			  curr_state=aut[curr_state][data_A[ii]];
			  if ( curr_state < 0 )
			    curr_state=0;
			}
		    }
		  for(size_t ii = 0; ii < aut.size(); ii++)
		    state_visit[ii]/=(len*NUM_EACH_this+0.0);

		  double NUMERATOR=0.0, DENOMINATOR=dist_entropy(phi_lambda); 		
		  for(unsigned int ii=0; ii < aut.size();++ii)
		    NUMERATOR += state_visit[ii]*dist_entropy(pit[ii]);

		  if (NUMERATOR > DENOMINATOR)
		    DENOMINATOR=0;
		  //		    cout << NUMERATOR << " " << DENOMINATOR << endl;

		  if (DENOMINATOR > 0)
		    {


		      this_gamma = 1 - NUMERATOR/DENOMINATOR;

		      /*
		      cout << this_gamma << " " 
			   << NUMERATOR << " " 
			   << DENOMINATOR  << " | " 
			   << vector<double> (phi_lambda) << " || " 
			   << vector<double> (pit[0])
			   << endl;
		      */
		      this_error *= (NUMERATOR/DENOMINATOR);
		    }
		  if (this_error < combined_error)
		    {
		      G_mc_.set_aut(aut);
		      G_mc_.set_pit(pit);
		      Xgamma = this_gamma;
		      eps=eps_vec[i];
		      annihilation_error=N;
		      combined_error = this_error;
		    }
		}
	      else
		{
		  double N=0.0;
		  for (int numeach=0; numeach<NUM_EACH_this;++numeach)
		    {
		      PFSA G;
		      G.set_aut(aut);
		      G.set_pit(pit);
		      Symbolic_string_ Ss( ~(Symbolic_string_ (data_B,alphabet_B))
					   + !(Symbolic_string_ (data_A,alphabet_A) *G));
		      //cout << Ss << endl;
			  
			  // Yi: modify here for SAE-free version 
		      Ss.get_norm(EVALUATION_DEPTH);
		      N+=Ss.norm;
		    }
		  N /= (NUM_EACH_this+0.0); /*Measure of global predictability*/

		  double LPred=0.0; /* Local predictability */
		  for (unsigned int pstate=0; pstate<pit.size();++pstate)
		    for (symbol psym(0); psym < pit[0].size(); ++psym)
		      if (pit[pstate][psym] > 0.0)
			LPred -= pit[pstate][psym]*log(pit[pstate][psym]);

		  LPred /= (pit.size()+0.0);

		  double CERR = N*LPred;     
		  if (CERR < combined_error)
		    {
		      G_mc_.set_aut(aut);
		      G_mc_.set_pit(pit);

		      local_error = LPred;
		      annihilation_error=N;
		      combined_error = CERR;
		      eps=eps_vec[i];
		    }
		}
	    }  // close for aut.size() > 0 //
	}
    }
};
//------------------------------------
//------------------------------------
ostream& operator<< (ostream &out, genESeSS &X)
{
  if ((!X.get_mc().get_aut().empty()) && (!X.get_mc().get_pit().empty()))
    {
      pitilde pit=X.get_mc().get_pit();
      connx aut=X.get_mc().get_aut();

       out << "%PITILDE: size(" <<  pit.size()+0.0 << ")" <<  endl;
     out << "#PITILDE" <<  endl;
      for (pitilde::iterator itr=pit.begin();
	   itr!=pit.end();++itr)
	{
	  for (vector <double>::iterator itr2=itr->second.begin();
	       itr2!=itr->second.end();
	       ++itr2)
	    out << *itr2 << " " ;
	  out << endl;
	}
      out << "%CONNX: size(" << aut.size()+0.0 << ")" << endl;
      out << "#CONNX" << endl;
      for (connx::iterator itr=aut.begin();
	   itr!=aut.end();++itr)
	{
	  for (map_sym_state::iterator itr1=itr->second.begin();
	       itr1!=itr->second.end();++itr1)
	    out <<  (int) itr1->second << " " ;
	  out << endl;
	}
    }  
  return out;
}
//------------------------------------

gParam::gParam(string configfile,unsigned int EVALDEPTH)
{
  MIN_DEV_SYNC_SEARCH_=0.001;
  LOOP_COUNT_SYNC_SEARCH_=1;
  SYN_PREF_LENGTH_ = 4;
  EPS_MIN_=0.001;
  EPS_MAX_=0.05;
  EPS_STEP_=0.001;
  ANN_TEST_LEN_=100000;
  COL_PRINT=true;
  EVALUATION_DEPTH=EVALDEPTH;
  EXTREMELY_SHORT_=true;
  NUM_EACH=1;
  MIN_STATE_SIZE=0;
  USE_GAMMA=true;

  CONFIG cfg(configfile);
  cfg.set(EPS_MIN_,"EPS_MIN_");
  cfg.set(EPS_MAX_,"EPS_MAX_");
  cfg.set(EPS_STEP_,"EPS_STEP_");
  cfg.set(ANN_TEST_LEN_,"ANN_TEST_LEN_");
  cfg.set(COL_PRINT,"COL_PRINT");
  cfg.set(EVALUATION_DEPTH,"EVALUATION_DEPTH");
  cfg.set(SYN_PREF_LENGTH_,"SYN_PREF_LENGTH_");
  cfg.set(LOOP_COUNT_SYNC_SEARCH_,"LOOP_COUNT_SYNC_SEARCH_");
  cfg.set(MIN_DEV_SYNC_SEARCH_,"MIN_DEV_SYNC_SEARCH_");
  cfg.set(EXTREMELY_SHORT_,"EXTREMELY_SHORT_");
  cfg.set(NUM_EACH,"NUM_EACH");
  cfg.set(MIN_STATE_SIZE,"MIN_STATE_SIZE");
  cfg.set(USE_GAMMA,"USE_GAMMA");

}
//------------------------------------
//   SIMPLE STREAM SERVER
//------------------------------------
void error(const char *msg)
{
  perror(msg);
  exit(1);
}
//---------------------------------
//---------------------------------
ostream& operator << (ostream &out, list <symbol> S)
{
  for(list <symbol>::iterator itr=S.begin(); 
      itr!=S.end();
      ++itr)
    out << (int)*itr << "."; 
  return out;
}
//---------------------------------
//---------------------------------
const list <symbol> & server_::data()
{
  return data_;
};
//---------------------------------
//---------------------------------

void server_::clear()
{
  data_.clear(); //No mutex
};
//---------------------------------
//---------------------------------

void server_::add_data(list <symbol> newdata)
{
  boost::mutex::scoped_lock lock(io_mutex);
  data_.splice(data_.end(),newdata); //mutex required
  while (data_.size() > MAX_SIZE_)
    data_.pop_front(); // constant data length 
};
//---------------------------------
//---------------------------------

void data_poll_(server_* H)
{
  char buffer[5000];
  ofstream OUT("log");
  symbol tmp;
  while (1)
    {
      memset(buffer,0,5000);	      
      int num_bytes_read = read(H->newsockfd,
				buffer,
				4098);
      if ( num_bytes_read >= 0) 
	{
	  list <symbol> newdata;
	  // convert buffer to list of symbols
	  /*	  string sbuff="";
	  sbuff+=buffer;
	  for(unsigned int i=0; i < sbuff.length();++i)
	      newdata.push_back(symbol (atoi(sbuff.substr(i,1).c_str())));
	  */

	  
	  istringstream ss(buffer);
	  string s=ss.str();

	  if((s.erase(s.find(' '), 1)=="C") || (s.erase(s.find(' '), 1)=="c"))
	    H->clear();

	  while (ss.good() && !ss.eof())
	    if (ss >> tmp)
	      newdata.push_back(tmp);
	  
	  OUT << newdata << endl;
	  H->add_data(newdata); 
	}
      if(num_bytes_read<=0)
	{
	  perror("Recv failed");
	  H->newsockfd = accept(H->sockfd, 
				(struct sockaddr *) &H->cli_addr, 
				&H->clilen);
	}
    }
};
//---------------------------------
//---------------------------------

server_::server_(int portno, int maxsize)
{
  MAX_SIZE_= maxsize;
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) 
    error("ERROR opening socket");
  memset((char *) &serv_addr,0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(portno);
  if (::bind(sockfd, (struct sockaddr *) &serv_addr,
	     sizeof(serv_addr)) < 0) 
    error("ERROR on binding");

  listen(sockfd,5);
  clilen = sizeof(cli_addr);
  newsockfd = accept(sockfd, 
		     (struct sockaddr *) &cli_addr, 
		     &clilen);
  if (newsockfd < 0) 
    error("ERROR on accept");
  T_ = new thread(&data_poll_,this);
};
//---------------------------------
//   SIMPLE STREAM CLIENT
//---------------------------------
client_::client_(int portno, const char* hostaddress)
{
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) 
    error("ERROR opening socket");

  struct in_addr addr;
  inet_aton(hostaddress, &addr);
  server = gethostbyaddr(&addr,sizeof(addr),AF_INET);
  //  
  if (server == NULL) {
    server = gethostbyname(hostaddress);
    if (server == NULL) {
      fprintf(stderr,"ERROR, no such host\n");
      exit(0);
    }
  }

  /*
  memset((char *) &serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char *)server->h_addr, 
	(char *)&serv_addr.sin_addr.s_addr,
	server->h_length);
  serv_addr.sin_port = htons(portno);
  if (connect(sockfd,(struct sockaddr *) 
	      &serv_addr,sizeof(serv_addr)) < 0) 
    error("ERROR connecting");

  */

  memcpy((char*) &serv_addr.sin_addr, 
	 server->h_addr_list[0], server->h_length);

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(portno);

  if (connect(sockfd,(struct sockaddr *) 
	      &serv_addr,sizeof(serv_addr)) < 0) 
    error("ERROR connecting");



};
//---------------------------------
//---------------------------------
void client_::send(string message)
{
  if (write(sockfd,
	    (message+" ").c_str(),
	    message.length()+1) 
      < 0)
    error("ERROR writing to socket");
};
//---------------------------------



//------------------------------------

// DATA READER -----------------------

//------------------------------------
data_reader::data_reader(string sdata_file,
			 unsigned int SZ)
{
  SERVER_MODE=false;
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
   
  while (getline (DATA, line))
    {
      symbol_list_ data_tmp;
      string tmp;
      ss.clear();
      ss.str ("");
      ss << line;
      while (data.size() < SZ && ss.good())
	{
	  ss >> tmp;
	  data_tmp.push_back( symbol (atoi(tmp.c_str())));
	}
      data.push_back(data_tmp);
    }
  DATA.close();
}
//------------------------------------

data_reader::data_reader(string sdata_file)
{
  SERVER_MODE=false;
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
   
  while (getline (DATA, line))
    {
      symbol_list_ data_tmp;
      string tmp;
      ss.clear();
      ss.str ("");
      ss << line;
      while (ss.good())
	{
	  ss >> tmp;
	  data_tmp.push_back( symbol (atoi(tmp.c_str())));
	}
      data.push_back(data_tmp);
    }
  DATA.close();
}

//------------------------------------

data_reader::data_reader(string sdata_file,
			 string data_dir,
			 unsigned int SZ,
			 bool KEY_)
{
  SERVER_MODE=false;
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
  symbol sym;

  if (data_dir == "across")
    {
      int count=0;
      while (getline (DATA, line))
	{
	  string key;
	  symbol_list_ data_tmp;
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  if (KEY_)
	    ss >> key;
	  while (ss.good() && data_tmp.size() < SZ)
	    {
	      ss >> sym;
	      data_tmp.push_back(sym);
	    }
	  data.push_back(data_tmp);
	  if (KEY_)
	    keys_[key]=count++;
	}
    }
  else
    {
      while (getline (DATA, line)&& data.size() < SZ)
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  unsigned int count=0;
	  if (KEY_)
	    {
	      string key;
	      while (ss.good() && !ss.eof())
		{
		  ss >> key;
		  keys_[key]=count++;
		}
	      KEY_=false;
	    }
	  else
	    {
	      while (ss.good())
		{
		  ss >> sym;
		  if (data.size() <= count)
		    data.push_back(vector <symbol> (1,sym));
		  else
		    data[count].push_back(sym);
		  count++;
		}
	    }
	}
    }
}
//------------------------------------

data_reader::data_reader(string sdata_file,
			 string data_dir)
{
  SERVER_MODE=false;
  ifstream DATA(sdata_file.c_str());
  string line;
  stringstream ss;
  symbol sym;

  if (data_dir == "across")
    {
      while (getline (DATA, line))
	{
	  symbol_list_ data_tmp;
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  while (ss.good())
	    {
	      ss >> sym;
	      data_tmp.push_back(sym);
	    }
	  data.push_back(data_tmp);
	}
    }
  else
    {
      while (getline (DATA, line))
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  unsigned int count=0;

	  while (ss.good())
	    {
	      ss >> sym;
	      if (data.size() <= count)
		data.push_back(vector <symbol> (1,sym));
	      else
		data[count].push_back(sym);
	      count++;
	    }
	}
    }
}
//------------------------------------
data_reader::data_reader(string cdata_file, 
			 vector < double > partition,
			 unsigned int SZ)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;

  while (getline (DATA, line))
    {
      symbol_list_ data_tmp;
      ss.clear();
      ss.str ("");
      ss << line;
      while (data.size() < SZ && ss.good())
	{
	  ss >> data_pt;
	  symbol j(0);
	  for (j=0; j<partition.size();j++)
	    if (data_pt<partition[j])
	      break;
	  data_tmp.push_back(j);
	}
      data.push_back(data_tmp);
    }
}
//------------------------------------
data_reader::data_reader(string cdata_file, 
			 vector < double > partition)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;

  while (getline (DATA, line))
    {
      symbol_list_ data_tmp;
      ss.clear();
      ss.str ("");
      ss << line;
      while (ss.good())
	{
	  ss >> data_pt;
	  symbol j(0);
	  for (j=0; j<partition.size();j++)
	    if (data_pt<partition[j])
	      break;
	  data_tmp.push_back(j);
	}
      data.push_back(data_tmp);
    }
}
//------------------------------------
/*
data_reader::data_reader(string cdata_file,
			 string data_dir,
			 vector <double> partition,
			 unsigned int SZ,
			 bool KEY_)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  double data_pt;
  if (data_dir == "across")
    {
      int count=0;
      while (getline (DATA, line))
	{
	  symbol_list_ data_tmp;
	  string key;
	  stringstream ss(line);
	  if (KEY_)
	    ss >> key;
	  while ((ss >> data_pt) && (data_tmp.size() < SZ))
	    {
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data_tmp.push_back(j);
	    }

	    data.push_back(data_tmp);


	  if (KEY_)
	    keys_[key]=count++;
	}
    }
  else
    {
      while (getline (DATA, line))
	{
	  stringstream ss(line);
	  int count=0;
	  if (KEY_)
	    {
	      string key;
	      while (ss >> key)
		keys_[key]=count++;
	      KEY_=false;
	    }
	  else
	    {
	      while (ss >> data_pt)
		{
		  symbol j(0);
		  for (j=0; j<partition.size();j++)
		    if (data_pt<partition[j])
		      break; 

		  if ((int)data.size() <=  count)
		    data.push_back(vector <symbol> (1,j));
		  else
		    data[count].push_back(j);
		  count++;
		}
	    }
	  if(data[0].size() == SZ)
	    break;
	}
    }
}*/
//------------------------------------
data_reader::data_reader(string cdata_file,
			 string data_dir,
			 vector <double> partition,
			 unsigned int SZ,
			 bool KEY_,
			 bool DERIVATIVE)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  double data_pt;

  if (data_dir == "across")
    {
      double old_data_pt=0.0;
      int count=0;
      
      while (getline (DATA, line))
	{
	  symbol_list_ data_tmp;
	  string key;
	  stringstream ss(line);
	  if (KEY_)
	    ss >> key;

	  while ((ss >> data_pt) && (data_tmp.size() < SZ))
	    {
	      if (DERIVATIVE)
		{
		  data_pt-=old_data_pt;
		  old_data_pt+=data_pt;
		}
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data_tmp.push_back(j);
	    }
	  data.push_back(data_tmp);


	  if (KEY_)
	    keys_[key]=count++;
	}
    }
  else
    {
      vector<double> old_data_pt;
      while (getline (DATA, line))
	{
	  stringstream ss(line);
	  int count=0;
	  if (KEY_)
	    {
	      string key;
	      while (ss >> key)
		keys_[key]=count++;
	      KEY_=false;
	    }
	  else
	    {
	      while (ss >> data_pt)
		{
		  if (DERIVATIVE)
		    {
		      if ((int)old_data_pt.size() <=  count)
			old_data_pt.push_back(0.0);

		    
		      data_pt-=old_data_pt[count];
		      old_data_pt[count]+=data_pt;
		    }

		  symbol j(0);
		  for (j=0; j<partition.size();j++)
		    if (data_pt<partition[j])
		      break; 

		  
		  if ((int)data.size() <=  count)
		    data.push_back(vector <symbol> (1,j));
		  else
		    data[count].push_back(j);
		  count++;
		}
	    }
	  if(data[0].size() == SZ)
	    break;
	}
    }
}
//------------------------------------

data_reader::data_reader(string cdata_file,
			 string data_dir,
			 vector < vector <double> > partition_vec,
			 unsigned int SZ,
			 bool KEY_)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;
  if (data_dir == "across")
    {
      int count=0;
      while (getline (DATA, line))
	{
	  if(count >= (int) partition_vec.size())
	    {
	      cout << "PARTITION VEC TOO SMALL" << endl;
	      exit(0);
	    }
	  vector <double> partition = partition_vec[count];
	  symbol_list_ data_tmp;
	  string key;
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  if (KEY_)
	    ss >> key;
	  while (ss.good() && data.size() < SZ)
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data_tmp.push_back(j);
	    }
	  data.push_back(data_tmp);
	  if (KEY_)
	    keys_[key]=count;
	  count++;
	}
    }
  else
    {
      while (getline (DATA, line) && data.size() < SZ)
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  int count=0;
	  if (KEY_)
	    {
	      string key;
	      while (ss.good() && !ss.eof())
		{
		  ss >> key;
		  keys_[key]=count++;
		}
	      KEY_=false;
	    }
	  else
	    {
	      while (ss.good() && !ss.eof())
		{
		  if(count >= (int) partition_vec.size())
		    {
		      cout << "PARTITION VEC TOO SMALL" << endl;
		      exit(0);
		    }
		  vector <double> partition = partition_vec[count];
		  ss >> data_pt;
		  symbol j(0);
		  for (j=0; j<partition.size();j++)
		    if (data_pt<partition[j])
		      break; 

		  if ((int)data.size() <=  count)
		    data.push_back(vector <symbol> (1,j));
		  else
		    data[count].push_back(j);
		  count++;
		}
	    }
	}
    }
}
//------------------------------------

data_reader::data_reader(string cdata_file,
			 string data_dir,
			 vector <double> partition)
{
  SERVER_MODE=false;
  ifstream DATA(cdata_file.c_str());
  string line;
  stringstream ss;
  double data_pt;
  if (data_dir == "across")
    {
      while (getline (DATA, line))
	{
	  symbol_list_ data_tmp;
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  while (ss.good())
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 
	      data_tmp.push_back(j);
	    }
	  data.push_back(data_tmp);
	}
    }
  else
    {
      while (getline (DATA, line))
	{
	  ss.clear();
	  ss.str ("");
	  ss << line;
	  unsigned int count=0;

	  while (ss.good())
	    {
	      ss >> data_pt;
	      symbol j(0);
	      for (j=0; j<partition.size();j++)
		if (data_pt<partition[j])
		  break; 

	      if (data.size() <= count)
		data.push_back(vector <symbol> (1,j));
	      else
		data[count].push_back(j);
	      count++;
	    }
	}
    }
}
//------------------------------------
//------------------------------------

map<string,int> data_reader::keys()
{
  if (keys_.empty())
    {
      for(unsigned int i=0;i<data.size();++i)
	{
	  char KEY[1024];
	  sprintf(KEY,"%d",i);
	  keys_[KEY] = i;
	}
    }
  return keys_;
}
//------------------------------------
//------------------------------------
//------------------------------------

data_reader::data_reader(int portnumber, int len)
{
  Server = new server_(portnumber,len);
  SERVER_MODE=true;
}
//------------------------------------

vector < Symbolic_string_ > data_reader::getsymbolic_string_vector()
{
  vector <Symbolic_string_> Svec;
  for (vector <symbol_list_>::iterator itr=data.begin();
       itr != data.end();
       ++itr)
    Svec.push_back(Symbolic_string_ (*itr));

  return Svec;  
}

//------------------------------------

Symbolic_string_  data_reader::getsymbolic_string()
{
  if (SERVER_MODE)
    {
      list <symbol> lData(Server->data());
      symbol_list_ tmp(lData.begin(),lData.end());
      return Symbolic_string_ (tmp);
    }
  else
    return Symbolic_string_ (data[0]);
}
//------------------------------------
symbol_list_  data_reader::getlist()
{
  if (SERVER_MODE)
    {
      list <symbol> lData(Server->data());
      return symbol_list_ (lData.begin(),lData.end());
    }
  else
    return data[0]; 
}


//------------------------------------
/**
   pretty print utilities
*/
//------------------------------------


//------------------------------------
//------------------------------------
ostream& operator << (ostream &out, connx &M)
{
  for(connx::iterator itr=M.begin();
      itr!=M.end();
      ++itr)
    {
      for(map_sym_state::iterator itr1=itr->second.begin();
	  itr1!=itr->second.end();
	  ++itr1)
	out  << itr1->second << " "; 
      out << endl;
    }
  return out;
}
//------------------------------------
//------------------------------------
ostream& operator << (ostream &out, symbol_list_ &s)
{
  for (unsigned int i=0; i<s.size(); i++)
    out <<  (s[i]) << sep;
  return out;
}
//------------------------------------
//------------------------------------

ostream& operator << (ostream &out, vector<double> v)
{
  for (unsigned int i=0; i<v.size(); i++)
    out <<  (v[i]) << " ";
  return out;
}

//------------------------------------
//------------------------------------
ostream& operator << (ostream &out,  
		      map < symbol, unsigned int > &M)
{
  for(map < symbol, 
	unsigned int >::iterator itr=M.begin(); 
      itr!=M.end();
      ++itr)
    out  <<  itr->second + 0.0 << " "; 

  return out;
}
//------------------------------------
//------------------------------------
ostream& operator << (ostream &out, phi_data_type_ &Phi)
{
  for(phi_data_type_::iterator itr=Phi.begin(); 
      itr!=Phi.end();
      ++itr)
    {
      symbol_list_ key(itr->first);
      out << key  << ": " << itr->second << endl; 
    }

  return out;
}
//------------------------------------
//------------------------------------
ostream& operator << (ostream &out, stoch_phi_data_type_ &sPhi)
{
  for(stoch_phi_data_type_::iterator itr=sPhi.begin(); 
      itr!=sPhi.end();
      ++itr)
    {
      symbol_list_ key(itr->first);
      out << key  << "-->" ; 
      for(vector <double>::iterator itr1=itr->second.begin();
	  itr1!=itr->second.end();
	  ++itr1)
	out << " " << *itr1 ;
      out << endl;
    }
  return out;
}
//------------------------------------


//--------------------------------
//--------------------------------
//--------------------------------
void parse_::ReplaceStringInPlace(string& subject,
			  const string& search,
                          const string& replace)
{
  size_t pos = 0;
  while((pos = subject.find(search,pos))
	!=string::npos)
    {
      subject.replace(pos,search.length(),
		      replace);
      pos+=replace.length();
    }
  return;
}

//--------------------------------
//--------------------------------
string parse_::spc(char* str)
{
  string tmp=str;
  for(set<string>::const_iterator itr
	=Operator.begin();
      itr!=Operator.end();++itr)
    ReplaceStringInPlace(tmp,*itr,
			 " "+*itr+" ");


  //sign check for double
  
  stringstream ss(tmp);
  vector<string> vec;
  string token, stmp="";
  while(ss>>token)
    vec.push_back(token);

  for(unsigned int i=0;i<vec.size()-1;++i)
    if(((vec[i]=="+")
	||(vec[i]=="-"))
       && (SCC_UTIL__::isOnlyDouble(vec[i+1].c_str())))
      stmp+=vec[i];
    else
      stmp+=vec[i]+" ";
  
  return stmp+vec.back();
}

//--------------------------------
//--------------------------------
set<string>& parse_::varnames()
{
  string tmp=output_;
  for(set<string>::const_iterator itr
	=Operator.begin();
      itr!=Operator.end();++itr)
    ReplaceStringInPlace(tmp,*itr,
			 " ");
  string var_;
  stringstream ss(tmp);
  while(ss>>var_)
    varnames_.insert(SCC_UTIL__::trim(var_));
  return varnames_;
}



//--------------------------------
//--------------------------------
string parse_::spc(string str)
{
  string tmp=str;
  for(set<string>::const_iterator itr
	=Operator.begin();
      itr!=Operator.end();++itr)
    ReplaceStringInPlace(tmp,*itr,
			 " "+*itr+" ");
  //sign check for double
  
  stringstream ss(tmp);
  vector<string> vec;
  string token, stmp="";
  while(ss>>token)
    vec.push_back(token);

  for(unsigned int i=0;i<vec.size()-1;++i)
    if(((vec[i]=="+")
	||(vec[i]=="-"))
       && (SCC_UTIL__::isOnlyDouble(vec[i+1].c_str())))
      stmp+=vec[i];
    else
      stmp+=vec[i]+" ";

  stmp=stmp+vec.back();
  
  tmp=stmp;
  stmp="";
  
  stringstream ss1(tmp);
  vec.clear();
  while(ss1>>token)
    vec.push_back(token);

  for(unsigned int i=0;i<vec.size();++i)
    if(((vec[i]=="+")
	||(vec[i]=="-"))
       && ((i==0)
	   ||((i>0)
	      &&(Operator.find(vec[i-1])!=Operator.end()))))
      {
	if(vec[i-1]=="(")
	  stmp+=" __FWN__ "+vec[i]+" ";
	else
	  {
	    if(vec[i-1]!=")")
	      MESSAGE("ERROR malformed expression, use parantheses");
	    else
	      stmp+=vec[i]+" ";
	  }
      }
    else
      stmp+=vec[i]+" ";

  return stmp;
}

//--------------------------------
//--------------------------------
parse_::parse_(char* s,set<string> operators,
	       map<string,
	       unsigned int> arity):Operator(operators),arity(arity)
{
  stringstream ss(spc(s));
  string token,str_="";
  vector<string> stack_,que_;
  
  while(ss>>token)  
    if(Operator.find(token)!=Operator.end())
      {
	if(token!=")")
	  stack_.push_back(token);
	else
	  {
	    while(stack_.back()!="(")
	      {
		que_.push_back(stack_.back());
		stack_.pop_back();
		if(stack_.empty())
		  cout << "ERROR: mismatched paran" << endl;
	      }
	    stack_.pop_back();
	  }
      }
    else
      que_.push_back(token);
  
  for(vector<string>::iterator itr=que_.begin();
      itr!=que_.end();
      ++itr)
    str_+=*itr+" ";
  for(vector<string>::reverse_iterator itr=stack_.rbegin();
      itr!=stack_.rend();
      ++itr)
    str_+=*itr+" ";

  output_=str_;
  return;
};

//--------------------------------
//--------------------------------
parse_::parse_(string s,set<string> operators,
	       map<string,
	       unsigned int> arity):Operator(operators),arity(arity)
{
  stringstream ss(spc(s));
  string token,str_="";
  vector<string> stack_,que_;
  
  while(ss>>token)  
    if(Operator.find(token)!=Operator.end())
      {
	if(token!=")")
	  stack_.push_back(token);
	else
	  {
	    while(stack_.back()!="(")
	      {
		que_.push_back(stack_.back());
		stack_.pop_back();
		if(stack_.empty())
		  cout << "ERROR: mismatched paran" << endl;
	      }
	    stack_.pop_back();
	  }
      } 
    else
      que_.push_back(token);
  
  for(vector<string>::iterator itr=que_.begin();
      itr!=que_.end();
      ++itr)
    str_+=*itr+" ";
  for(vector<string>::reverse_iterator itr=stack_.rbegin();
      itr!=stack_.rend();
      ++itr)
    str_+=*itr+" ";

  output_=str_;
  return;
};

//--------------------------------
//--------------------------------
ostream& operator << (ostream &out, parse_& s)
{
  out << s.output_;
  return out;
}

//--------------------------------
string parse_::get()
{
  return output_;
}

//--------------------------------
//--------------------------------

vector<string> parse_::get(string& expr,string plc)
{
  stringstream ss(expr);
  string token;
  vector<string> strvec;
  while(ss>>token)
    if(Operator.find(token)!=Operator.end())
      break;
    else
      strvec.push_back(token);

  string remstr="";
  vector<string> thisopr;
  thisopr.push_back(token); 

  getline(ss,remstr);
  map<string,unsigned int>::const_iterator token_itr
    =arity.find(token);
  
  if(token_itr->second==1)
    {
      thisopr.push_back(strvec.back());
      strvec.pop_back();
    }
  if(token_itr->second==2)
    {
      if(strvec.size()>=2)
	{
	  thisopr.push_back(strvec.back());
	  strvec.pop_back();
	  thisopr.push_back(strvec.back());
	  strvec.pop_back();	
	}
      else
	MESSAGE("ERROR not enough variables");
    }

  if(SCC_UTIL__::trim(remstr)=="")
    expr="";
  else
    {
      strvec.push_back(plc);
      string str__="";
      for(vector<string>::iterator itr=strvec.begin();
	  itr!=strvec.end();
	  ++itr)
	str__+=*itr+" ";
      expr=str__+remstr;
    }
  return thisopr;
};




//------------------------------------
double dist_entropy(vector <double> p)
{
  double e=0.0;
  for (unsigned int i=0; i <p.size();++i)
    if (p[i] > 0.0)
      e -= p[i]*log2(p[i]);
  return e;
}; 
//------------------------------------
//------------------------------------
double vec_distance(vector <double> v0, vector <double> v1)
{
  double distance=0.0;
  if(v0.size() != v1.size())
    return 100.0;
  for(unsigned int i=0; i < v0.size();++i)
    if(fabs(v0[i]-v1[i])>distance)
      distance = fabs(v0[i]-v1[i]);
  return distance;		
}
//------------------------------------

void SCC_UTIL__::print_status(float progress)
{
  string str="";
  char buff[1024];
  sprintf(buff,"%c[%d;%d;%dm",27,0,32,40);
  str+=buff;
  string str1="";
  sprintf(buff,"%c[%dm",27,0);
  str1+=buff;


  int barWidth = 40;

  std::cout << str<< "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r" << str1;

  std::cout.flush();
}
//------------------------------------
//------------------------------------

double SCC_INNER_PROD__::mult(const vector<double>& a,
			      const vector<double>& b)
{
  double ZERO=1e-15;
  if (a.size()!=b.size())
    MESSAGE("ERROR: SIZE MISMATCH");
  
  double s=0.0;
  for(unsigned int i=0;i<a.size()-1;++i)
    s+=log(a[i]/(a[i+1]+ZERO))*log(b[i]/(b[i+1]+ZERO));
  return s;
};


//------------------------------------
void SCC_UTIL__::_MESSAGE(string m,const char *caller)
{
  
  cout << boost::to_upper_copy(m)
       << " in function "
       << std::string(caller) << endl;
  if(m.find("ERROR") != std::string::npos)
    exit(0);

};


//---------------------------------------------
bool SCC_UTIL__::removeSubstrs(string& s,
                   const string& p) {
  size_t n = p.length();
  bool FOUND=false;
  for (size_t i = s.find(p);
       i != string::npos;
       i = s.find(p))
    {
      FOUND = true;
      s.erase(i, n);
    }
  return FOUND;
};

//---------------------------------------------
string SCC_UTIL__::removeSubstrs__(string& s) {
  s.erase( std::remove_if( s.begin(), s.end(), ::isspace ), s.end() );
  return s;
};

//---------------------------------------------
void SCC_UTIL__::ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
  size_t pos = 0;
  while((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

//---------------------------------------------
string SCC_UTIL__::trim(const string& str,
	    const string& whitespace)
{
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == string::npos)
    return ""; // no content

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;

  return str.substr(strBegin, strRange);
};
//---------------------------------------------
bool SCC_UTIL__::isOnlyDouble(const char* str)
{
    char* endptr = 0;
    strtod(str, &endptr);

    if(*endptr != '\0' || endptr == str)
        return false;
    return true;
}

//---------------------------------------------
PFSA SCC_UTIL__::read_mc(string filename, string TYPE__)
{

  connx aut;
  pitilde pit, Xpit;
  map < state, map < symbol, double > > pitilde_map;

  CONFIG modfile(filename);
  modfile.set_map<state,symbol,state>(aut,"CONNX");
  modfile.set_map<state,symbol,double>(pitilde_map,"PITILDE");


  PFSA *G;
  state c_state=0;
  for ( map < state, map < symbol, 
	  double > >::iterator iti=pitilde_map.begin();
	iti != pitilde_map.end();
	iti++)
    {
      vector <double> vec_tmp;
      for (map <symbol, double>::iterator itj=iti->second.begin();
	   itj != iti->second.end();
	   itj++)
	vec_tmp.push_back(itj->second);
      pit[c_state++] = vec_tmp;
    }

  if(TYPE__=="XPFSA") 
    {
      pitilde pit2;
      for (state  j=0; j < (int)aut.size();++j)
	pit2[j]= vector <double> (aut[0].size(),1.0/(aut[0].size()+0.0));
      G = new PFSA(pit2,aut);
    }
  else
    G = new PFSA (pit,aut);
  G->set_Xpit(pit);

  return *G;
};
//---------------------------------------------

PFSA SCC_UTIL__::FWN(unsigned int alph_)
{
  connx aut;
  pitilde pit;

  for(unsigned int i=0;i<alph_;++i)
    aut[0][symbol (i)]=0;

  pit[0]=vector<double>(alph_,1/(alph_+0.0));

  PFSA *G;
  G = new PFSA (pit,aut);
  return *G;

};

//---------------------------------------------

PFSA SCC_UTIL__::generate_mc(unsigned int alphabet,
			     unsigned int numstates,
			     string MC_TYPE)
{
  connx aut;
  if(numstates<=1)
    return SCC_UTIL__::FWN(alphabet);
  
  if (MC_TYPE=="M")
    {
      aut.clear();
      state st0=0;
      while(st0<(int)numstates)
	{
	  symbol sym(0);
	  for(state st=0;
	      st<(int)numstates;
	      ++st)
	    {
	      aut[st0][sym]=st;
	      sym++;
	      if(sym == symbol(alphabet))
		{
		  sym=symbol(0);
		  st0++;
		}
	    }
	}
    }
  
  if (MC_TYPE=="T")
    {
      aut.clear();
      state st0=0;
      while(st0<(int)numstates)
	{
	  state st=++st0;
	  for(symbol sym(0);
	      sym<alphabet;
	      ++sym)
	    {
	      aut[st0-1][sym]=st;
	      if(st==(int)numstates)
		{
		  aut[st0-1][sym]=0;
		  st=0;
		}
	      st++;
	    }
	}
    }
  
  if (MC_TYPE=="S")
    {
      aut.clear();
      for(state st0=0;
	  st0<(int)numstates;
	  ++st0)
	{
	  state st=st0;
	  for(symbol sym(0);
	      sym<alphabet;
	      ++sym)
	    {
	      aut[st0][sym]=st;
	      if(st==(int)numstates)
		{
		  aut[st0][sym]=0;
		  st=0;
		}
	      st++;
	    }
	}
    }


  pitilde pit;
  list<double> pvec;
  if (numstates >= alphabet)
    {
      map<symbol,list<double> > H;
      map<symbol,vector<double> > V;
      for(state st=0;st<(int)numstates;++st)
	pvec.push_back((st+1)/(numstates+1.0));

      for(symbol sym(0);sym<alphabet;++sym)
	{
	  pvec.push_back(pvec.front());
	  pvec.pop_front();
	  H[sym]=pvec;
	}
      
      for(symbol sym(0);sym<alphabet;++sym)
	{
	  list<double> tmplist(H[sym]);
	  for(state st=0;st<(int)numstates;++st)
	    {
	      V[sym].push_back(tmplist.front());
	      tmplist.pop_front();
	    }
	}
      
      for(state st=0;st<(int)numstates;++st)
	{
	  vector<double> row;
	  double s=0.0;
	  for(symbol sym(0);sym<alphabet;++sym)
	    {
	      row.push_back(V[sym][st]);
	      s+=row.back();
	    }
	  if(s>0.0)
	    for(unsigned int j=0;j<alphabet;++j)
	      row[j]/=s;
	  pit[st]=row;
	}
    }
  else
    {      
      map<state,list<double> > H;
      double s=0.0;
      for(unsigned int j=0;j<alphabet;++j)
	{
	  pvec.push_back((j+1)/(alphabet+1.0));
	  s+=pvec.back();
	}
      
      for(state st=0;st<(int)numstates;++st)
	{
	  copy(pvec.begin(),pvec.end(),
	       back_inserter(pit[st]));
	  pvec.push_back(pvec.front());
	  pvec.pop_front();
	  for(unsigned int j=0;
	      j<alphabet;++j)
	    pit[st][j]/=s;
	}
      
    }
  return PFSA (pit,aut);
};

