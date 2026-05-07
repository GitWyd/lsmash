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
#ifndef __CONFIG_H
#define __CONFIG_H

#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <map>

using namespace std;

class CONFIG
{
  string cfgfile;

  vector<string> cheap_tokenise(string const& );
  bool strbeg(string, string);

  template <class T>
    bool any_cast(T& t, const std::string& s)
    {
      std::istringstream iss(s);
      return !(iss  >> t).fail();
    };

  /*

  int var_int;
  unsigned int var_unsigned _int;
  double var_double;
  string var_string;
  long int var_long_int;
  long double var_long_double;
  float var_float;
  char var_char;
  unsigned char var_unsigned_char;
  bool var_bool;
  */
 public:

  CONFIG(string);

  //------------------------------------------------------------
  template <class T>
    bool set(T & Varname, string Fname)
    {
      string prefFname="#";
      prefFname+=Fname;
      string line;
      ifstream cfg(cfgfile.c_str());

      bool status=true;
      if (cfg.is_open())
	{
	  /**
	     The following map is platform and compiler dependent
	     There is apparently no portable way of 
	     getting unmangled names from typeid
	  */
	  map <type_index,string> typeMap;
	  typeMap[type_index(typeid(int))]="int";
	  typeMap[type_index(typeid(unsigned int))]="unsigned int";
	  typeMap[type_index(typeid(double))]="double";
	  typeMap[type_index(typeid(string))]="string";
	  //	  typeMap["Ss"]="string";
	  typeMap[type_index(typeid(long int))]="long int";
	  typeMap[type_index(typeid(long double))]="long double";
	  typeMap[type_index(typeid(float))]="float";
	  typeMap[type_index(typeid(char))]="char";
	  typeMap[type_index(typeid(unsigned char))]="unsigned char";
	  typeMap[type_index(typeid(bool))]="bool";

	  string Vartype=typeMap[type_index(typeid(Varname))];


	  //if(decltype (var_string) == decltype (Varname))
	  
	  if (Vartype=="short"
	      || Vartype=="int" 
	      || Vartype=="unsigned int" 
	      || Vartype=="double" 
	      || Vartype=="float" 
	      || Vartype=="long int" 
	      || Vartype=="long double"
	      || Vartype=="unsigned char" 
	      || Vartype=="string"
	      || Vartype=="bool"
	      || Vartype=="char" )
	    {
	      while (getline(cfg,line))
		{
		  vector <string>vline=cheap_tokenise(line);
		  if ((vline.size() > 0) && ( vline[0]==prefFname ))
		    {
		      string line2;
		      while(getline(cfg,line2))
			{
			  vector <string>vline2=cheap_tokenise(line2);
			  if (vline2.size() > 0) 
			    if (!strbeg(vline2[0],"%"))
			      {
				status=any_cast(Varname,vline2[0]);
				break;
			      }
			}
		    }
		}
	    }
	  else
	    {
	      cout << "xxx " <<	typeid(Varname).name() << endl;
	    }
	  cfg.close();
	}
      return status;
    };
  //------------------------------------------------------------
  template <class T>
    bool set_vector(vector <T> & Varname, 
		    string Fname)
    {
      string prefFname="#";
      prefFname+=Fname;
      string line;
      ifstream cfg(cfgfile.c_str());
      bool status=true;
      if (cfg.is_open())
	{
	  if (!Varname.empty())
	    Varname.clear();
	  bool read=false;
	  while ( getline(cfg,line) &&( !read) )
	    {
	      vector <string>vline=cheap_tokenise(line);
	      if ((vline.size() > 0) && ( vline[0]==prefFname ))
		{
		  string line2;
		  while(getline(cfg,line2))
		    {
		      vector <string>vline2=cheap_tokenise(line2);
		      if (vline2.size() > 0) 
			if (!strbeg(vline2[0],"%"))
			  {
			    for (unsigned int index=0;index<vline2.size();index++)
			      {
				if (!strbeg(vline2[0],"#"))
				  {
				    T tmp;
				    status=any_cast(tmp,vline2[index]);
				    Varname.push_back(tmp);
				  }
			      }
			    read=true;
			    break;
			  }
		    }
		}
	    }
	  cfg.close();
	}	
      return status;
    };
  //------------------------------------------------------------
  //------------------------------------------------------------
  template <class T>
    bool set_column_vector(vector <T> & Varname, 
			   string Fname)
    {
      string prefFname="#";
      prefFname+=Fname;
      string line;
      ifstream cfg(cfgfile.c_str());
      bool status=true;
      if (cfg.is_open())
	{
	  if (!Varname.empty())
	    Varname.clear();
	  bool read=false;
	  while ( getline(cfg,line) &&( !read) )
	    {
	      vector <string>vline=cheap_tokenise(line);
	      if ((vline.size() > 0) && ( vline[0]==prefFname ))
		{
		  string line2;
		  while(getline(cfg,line2))
		    {
		      //	vector <string>vline2=cheap_tokenise(line2);
		      if (line2.size() > 0) 
			if (!strbeg(line2,"%"))
			  {
			    if (!strbeg(line2,"#"))
			      {
				size_t found;
				found=line2.find("%");
				if (found!=string::npos)
				  line2=line2.substr(0,found);

				T tmp;
				status=any_cast(tmp,line2);
				Varname.push_back(tmp);
			      }
			    else
			      {
				read=true;
				break;
			      }
			  }
		    }
		}
	    }
	  cfg.close();
	}	
      return status;
    };
  //------------------------------------------------------------

  template <class Ti, class Tj, class T>
    bool set_map(map <Ti, map <Tj,T> > & Varname, 
		 string Fname)
  {
    string prefFname="#";
    prefFname+=Fname;
    string line;
    ifstream cfg(cfgfile.c_str());
    bool status=true;
    if (cfg.is_open())
      {
	Ti row_count=0;
	bool read=false;
	while ( getline(cfg,line) &&( !read) )
	  {
	    vector <string>vline=cheap_tokenise(line);
	    if ((vline.size() > 0) && ( vline[0]==prefFname ))
	      {
		string line2;
		while(getline(cfg,line2))
		  {
		    vector <string>vline2=cheap_tokenise(line2);
		    if (vline2.size() > 0) 
		      if (!strbeg(vline2[0],"%"))
			{
			  map <Tj,T> Var_tmp;
			  bool valset=false;
			  Tj index;
			  for (index=0;index<vline2.size();index++)
			    {
			      if (!strbeg(vline2[0],"#"))
				{
				  T tmp;
				  status=any_cast(tmp,vline2[index]);
				  Var_tmp[index]=tmp;
				  valset=true;
				}
			    }
			  if (valset)
			    Varname[row_count++] = Var_tmp;
			}
		    if (vline2.size() > 0) 
		      if  (strbeg(vline2[0],"#"))
			{
			  read=true;
			  break;
			}
		  }
	      }
	  }
	cfg.close();
      }	
    return status;
  };
  //------------------------------------------------------------
  //------------------------------------------------------------

  template <class Tj>
    bool set_vector_vector(vector < vector <Tj> > & Varname, 
		 string Fname)
  {
    string prefFname="#";
    prefFname+=Fname;
    string line;
    ifstream cfg(cfgfile.c_str());
    bool status=true;
    if (cfg.is_open())
      {
	bool read=false;
	while ( getline(cfg,line) &&( !read) )
	  {
	    vector <string>vline=cheap_tokenise(line);
	    if ((vline.size() > 0) && ( vline[0]==prefFname ))
	      {
		string line2;
		while(getline(cfg,line2))
		  {
		    vector <string>vline2=cheap_tokenise(line2);
		    if (vline2.size() > 0) 
		      if (!strbeg(vline2[0],"%"))
			{
			  vector <Tj> Var_tmp;
			  bool valset=false;
			  Tj index;
			  for (index=0;index<vline2.size();index++)
			    {
			      if (!strbeg(vline2[0],"#"))
				{
				  Tj tmp;
				  status=any_cast(tmp,vline2[index]);
				  Var_tmp.push_back(tmp);
				  valset=true;
				}
			    }
			  if (valset)
			    Varname.push_back(Var_tmp);
			}
		    if (vline2.size() > 0) 
		      if  (strbeg(vline2[0],"#"))
			{
			  read=true;
			  break;
			}
		  }
	      }
	  }
	cfg.close();
      }	
    return status;
  };
  //------------------------------------------------------------

};
//------------------------------------------------------------
#endif
