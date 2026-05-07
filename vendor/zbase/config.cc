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
#include "config.h"

using namespace std;

vector<string> CONFIG::cheap_tokenise(string const& input)
{
  istringstream str(input);
  istream_iterator<string> cur(str), end;
  return vector<string>(cur, end);
}

bool CONFIG::strbeg(string s, string frag)
{
  unsigned int lfrag=frag.length();
  string sbstr=s.substr(0,lfrag);
  if (sbstr==frag)
    return true;
  return false;
}

CONFIG::CONFIG(string s)
{
  cfgfile=s;
}

