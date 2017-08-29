#!/usr/bin/env python

class Individual :
    
    def __init__ (self, activation_list, fitness) :
        self.activation_list = activation_list
        self.fitness = fitness
    
    def __eq__ (self, other) :
        if isinstance(other, Individual) :
            return self.activation_list == other.activation_list
        elif isinstance(other, (int, float)) :
            return self.fitness == other
        else :
            return NotImplemented
        
    def __le__ (self, other) :
        if isinstance(other, Individual) :
            return self.fitness <= other.fitness
        elif isinstance(other, (int, float)) :
            return self.fitness <= other
        else :
            return NotImplemented
    
    def __ge__ (self, other) :
        if isinstance(other, Individual) :
            return self.fitness >= other.fitness
        elif isinstance(other, (int, float)) :
            return self.fitness >= other
        else :
            return NotImplemented
    
    def __lt__ (self, other) :
        if isinstance(other, Individual) :
            return self.fitness < other.fitness
        elif isinstance(other, (int, float)) :
            return self.fitness < other
        else :
            return NotImplemented
    
    def __gt__ (self, other) :
        if isinstance(other, Individual) :
            return self.fitness > other.fitness
        elif isinstance(other, (int, float)) :
            return self.fitness > other
        else :
            return NotImplemented
    
    def __str__ (self) :
        return str(self.fitness)


class MO_Individual :
    
    def __init__ (self, activation_list, fitness_list) :
        self.activation_list = activation_list
        self.fitness = fitness_list
        self.strength = 0
        self.wimpiness = 0
        self.dominated_by = []
    
    def __eq__ (self, other) :
        if isinstance(other, MO_Individual) :
            return self.activation_list == other.activation_list
        elif isinstance(other, (int, float)) :
            return self.wimpiness == other
        else :
            return self.activation_list == other
        
    def __le__ (self, other) :
        if isinstance(other, MO_Individual) :
            return self.wimpiness <= other.wimpiness
        elif isinstance(other, (int, float)) :
            return self.wimpiness <= other
        else :
            return NotImplemented
    
    def __ge__ (self, other) :
        if isinstance(other, MO_Individual) :
            return self.wimpiness >= other.wimpiness
        elif isinstance(other, (int, float)) :
            return self.wimpiness >= other
        else :
            return NotImplemented
    
    def __lt__ (self, other) :
        if isinstance(other, MO_Individual) :
            return self.wimpiness < other.wimpiness
        elif isinstance(other, (int, float)) :
            return self.wimpiness < other
        else :
            return NotImplemented
    
    def __gt__ (self, other) :
        if isinstance(other, MO_Individual) :
            return self.wimpiness > other.wimpiness
        elif isinstance(other, (int, float)) :
            return self.wimpiness > other
        else :
            return NotImplemented
    
    def __str__ (self) :
        return str(self.fitness)


# each instance of this class stores a single fittness rule
class Fitness_Rule :  
    # takes a string which containes the rule and builds the rule
    def __init__(self, line) :
        self.option_list = []
        line_list = line.split('#')
        for element in line_list :
            if element.count(':') == 0 :
                self.option_list.append(element)
            else :
                self.option_list.append(element[:element.find(':')])
                self.fitnes_value = float(element[element.find(':')+2:])    
                
    # print overload
    def __str__(self) :
        out = ""
        for element in self.option_list :
            out += element
            out += " ** "
        out = out[:-3]
        out += " : " + str(self.fitnes_value)
        return out
        
    # takes an option dictionariy and returns the cost of this rule
    def get_partial_fitness(self, option_activation) :
        applicable = 1
        for it in self.option_list :
            applicable = applicable and option_activation[it]
        return applicable * self.fitnes_value        

    
# contains all fittness rules
class Fitness_Modell :    
    
    # takes a list containing fittness rules and builds the rule list
    def __init__(self) :
        self.fittness_rule_list = []
        
        
    # parse names to ints to use them as an index later
    def parse_names_to_index(self, dimacs_input) : 
        
        name_index_dict = dict()
        
        dimacs_input = [x.strip() for x in dimacs_input]
        for line in dimacs_input :
            # fill constraint list 
            if len(line) > 0 and line[0] == 'c' :
                line_list = line.split(' ')
                if len(line_list) == 3 :
                    name_index_dict[line_list[2]] = int(line_list[1].strip('$'))
        
        for rule in self.fittness_rule_list :
            for i in range(0, len(rule.option_list)) :
                if not rule.option_list[i] == 'root' :
                    rule.option_list[i] = name_index_dict[rule.option_list[i]]
                else :
                    rule.option_list[i] = 0
                    
                    
    # takes a string containing fittness rules and adds them to the rule list
    def add_fitness_rules(self, fittness_input) :
        for line in fittness_input :
            self.fittness_rule_list.append(Fitness_Rule(line))        

                        
    # print overload
    def __str__(self) :
        out = ""
        for element in self.fittness_rule_list :
            out += element.__str__() + "\n"
        out = out[:-2]
        return out
    
    # takes an option activation dictionary and calculates the fitness(or cost) value
    def calculate_fitness(self, option_activation) : 
        fittness = 0.0
        for element in self.fittness_rule_list :
            fittness += element.get_partial_fitness(option_activation)
        return 1 / fittness
    
    def calculate_cost(self, option_activation) : 
        fittness = 0.0
        for element in self.fittness_rule_list :
            fittness += element.get_partial_fitness(option_activation)
        return fittness


class MO_Fitness_Modell :
    def __init__(self, fitness_modell_list) :
        self.fitness_modell_list = fitness_modell_list    
    
    def calculate_fitness(self, activation) :
        fitnes_list = []
        for it in self.fitness_modell_list :
            fitnes_list.append(it.calculate_fitness(activation))
        return fitnes_list
    
    def calculate_cost(self, activation) :
        cost_list = []
        for it in self.fitness_modell_list :
            cost_list.append(it.calculate_cost(activation))
        return cost_list


class MO_Population :
    def __init__(self, mo_fitness_modell, constraint_modell) :
        self.helper = Helper_Functions()
        self.fitness_model = mo_fitness_modell
        self.constraint_modell = constraint_modell
        self.population = []
        self.best = []
        
    def update_strength(self) :
        for i in self.population :
            i.dominated_by = []
            
        for i in range(0, len(self.population)) :
            self.population[i].strength = 0
            for j in range(0, len(self.population)) :
                if not(i == j) :
                    dominated = False
                    for objective in range(0, len(self.population[j].fitness)) :
                        if self.population[i].fitness[objective] < self.population[j].fitness[objective] :
                            dominated = False
                            break
                        if self.population[i].fitness[objective] > self.population[j].fitness[objective] :
                            self.population[j].dominated_by.append(i)
                            dominated = True
                    if dominated :
                        self.population[i].strength += 1
    
    def update_wimpiness(self) :
        for it in self.population :
            it.wimpiness = 0
            for j in it.dominated_by :
                it.wimpiness += self.population[j].strength
    
    
    def generate_population(self, popsize) :
        population_tmp = pycosat_init(self.constraint_modell, popsize)
        
        # appends elements, that are not already in p and assess fittness
        for ind in population_tmp :
            if not ind in self.population :
                # the pycosat individuals are very close, so we make big mutations in the begining
                ind = self.helper.mutate(ind, self.constraint_modell, 1)                
                self.population.append(MO_Individual(ind, self.fitness_model.calculate_fitness(ind))) 
                
        self.update_strength()
        self.update_wimpiness()
        
        self.population.sort()
        
    def __str__(self) :
        out = ""
        for it in self.population :
            out += (str(it.fitness) + "\n")
        return out
    
    
    
    def learn(self, iterations, u, l):
        self.population.sort()
        del(self.population[u:])
        print("before learning")
        print(self)
        last = 0
        for counter in range(0, iterations) :
            for i in range(0,l) :
                tmp = self.helper.mutate(self.population[i % l].activation_list, self.constraint_modell, 1)
                self.population.append(MO_Individual(tmp, self.fitness_model.calculate_fitness(tmp)))
            self.population.sort()
            del(self.population[u:])
            if counter >= (last + 5) :
                print(str(counter) + " rounds")
                print(self)
            
        print("after learning")
        print(self)


class MO_Wrapper :
    def __init__(self, path_to_folder, project_name, feauture_list, interaction_list, posize = 100) :
        from os import listdir
        from os.path import isfile, join
        
              
        self.population = []
        
        constraint_model = Constraint_Model()      
        fittness_model_list = []
        
        
        # -------- Fitness Model -----------
        for i in range(0, len(feauture_list)) :
            fittness_model_list.append(Fitness_Modell())
        
        for i in range(0, len(feauture_list)) :
            files = [join(path_to_folder, f) for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
            tmp_dimacs = []
            for file in files:
                if file.count(project_name) > 0 and (file.count(feauture_list[i]) > 0 or file.count(interaction_list[i]) > 0) :
                    with open(file, 'r') as f:
                        fittness_model_list[i].add_fitness_rules(f.readlines())
                        
                        
                        
        # ------ Constraint Model ---------
        files = [join(path_to_folder, f) for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
        tmp_dimacs = []

        for file in files:
            if file.count(project_name) > 0 and file.count('.xml') > 0 :
                tmp_dimacs = parse_xml_to_dimacs(file)
                self.constraint_model.build_model(tmp_dimacs)
            if file.count(project_name) > 0 and file.count('.dimacs') > 0 :
                with open(file, 'r') as f:
                    tmp_dimacs = f.readlines()
                    constraint_model.build_model(tmp_dimacs)   
                    
                                    
        # analyze constraints
        constraint_model.build_global_tabu_list()
        
        # parse fittnes rule names to ints
        for it in fittness_model_list :
            it.parse_names_to_index(tmp_dimacs)
        
        mo_fitness_modell = MO_Fitness_Modell(fittness_model_list)
        
        self.population = MO_Population(mo_fitness_modell, constraint_model)
        self.population.generate_population(posize)
        print(self.population)


# an instance of this class stores all constrains
# TODO: optimize validity check
class Constraint_Model :
        
    def __init__(self) :
        self.constraint_list = []
        self.simplified_constraint_list = []
        self.global_tabu_list = []
        self.mandatory_activation = []
                 
    # input line(string) list
    def build_model(self, dimacs_input) :
        self.constraint_list = []
        self.global_tabu_list = []
        self.mandatory_activation = []
        tmp_constraint = []
        
        dimacs_input = [x.strip() for x in dimacs_input]
        for line in dimacs_input :
            # fill constraint list 
            if len(line) > 0 and line[0] != 'c' and line[0] != 'p' :
                line_list = line.split(' ')
                for element in line_list :
                    if element == '0' and len(element) > 0 :
                        self.constraint_list.append(tmp_constraint)
                        tmp_constraint = []
                    elif len(element) > 0 :
                        tmp_constraint.append(int(element))      
            # initialize tabu list and mandatory activation
            elif len(line) > 0 and line[0] == 'p' :
                line_list = line.split(' ')
                self.global_tabu_list = [False] * (int(line_list[2]) + 1)
                self.mandatory_activation = [None] * (int(line_list[2]) + 1)

        self.simplified_constraint_list = self.constraint_list
                
                
    # print function            
    def __str__ (self) :
        out = "Constraints:\n"
        for a in self.constraint_list :
            out += str(a) + "-"+ str(len(a)) + ", "
        
        out += "\n\nGLobal Tabu List:\n"
        out += str(self.global_tabu_list)
        
        out += "\n\nMandatory Activation List:\n"
        out += str(self.mandatory_activation)
           
        return out
    
    def get_constraints(self) :
        out = []
        for a in self.constraint_list :
            out.append(a)
        return out
        
    
    # TO DO: delete more useless constraints
    # builds the global tabu list and sets the activation values 
    def build_global_tabu_list(self) :
        new_constraint_list = []
        for constraint in self.constraint_list :  
            if len(constraint) == 1 :       
                self.global_tabu_list[abs(constraint[0])] = True      
                self.mandatory_activation[abs(constraint[0])] = constraint[0] > 0    
            else :
                new_constraint_list.append(constraint)
                
        self.simplified_constraint_list = new_constraint_list
    
    
    # TODO: optimize !!!!!!
    def get_violated_variables (self, activation_arry) :
        violated_variables = []
        for constraint in self.simplified_constraint_list :
            if not self.check_partial_validity(constraint, activation_arry) :
                for v in constraint :
                    if abs(v) not in violated_variables :
                        violated_variables.append(abs(v))
        violated_variables.sort()
        return violated_variables
    
    
    # returns the first violated constraint
    def get_violated_constraint(self, activation_arry) :
        for constraint in self.simplified_constraint_list :
            if not self.check_partial_validity(constraint, activation_arry) :
                return constraint
        return []   
    
    # TODO: optimize !!!!!!    
    def check_validity (self, activation_arry) :
        for constraint in self.simplified_constraint_list :
            if not self.check_partial_validity(constraint, activation_arry) :
                return False
        return True
    
    # TODO: optimize
    def check_partial_validity (self, constraint, activation_arry) :
        for x in constraint :
            if x > 0 :
                if activation_arry[x] :
                    return True
            else :
                 if not activation_arry[x * (-1)] :
                    return True   
        return False



def linear_regression(n) :
    import itertools
    combinations = list(map(list, itertools.product([0, 1], repeat=n)))
    return(combinations)

def parse_xml_to_dimacs(file):
    var_map = [] #

    final_string = "" #
    variables_string = "" #
    p_line = "p cnf " #
    clauses = "" #
    current_options = [] # 
    optional_false_elements = []
    clauses_2D = []

    from lxml import etree
    xml_rules = etree.parse(open(file, 'r'))
    root = xml_rules.getroot()

    # first iteration to generate the dimacs variables
    for child in root :
        for subchild in child :
            for element in subchild : 
                if element.tag == 'name':
                    var_map.append(element.text)

    # generate variables
    for el in var_map:
        variables_string += "c " + str(1+var_map.index(el)) + " " + el + "\n"

    # generate rules
    for child in root :
        for subchild in child :
            for element in subchild :

                # temporary save current element and excluded options
                if element.tag == 'name':
                    current_options.append(element.text)
                if element.tag == 'excludedOptions' :
                    for option in element :
                        current_options.append(option.text)

                # clear temp if 'otional' tag is set to true
                if element.tag == 'optional' :
                    if element.text != 'False' :
                        current_options = []

            # if there are choices to do generate all combinations at once
            if len(current_options) > 0 :
                ###########################################################################
                # adding all combinations of rules if they are not already in 
                ###########################################################################

                if len(current_options) == 1 :
                    for element in current_options:
                        tmp=[]
                        tmp.append(1+var_map.index(element))
                        clauses_2D.append(tmp)

                if len(current_options) > 1 :
                    ####################################################current_options = current_options.sort()
                    combinations = linear_regression(len(current_options))
                    for lst in combinations :
                        if not lst.count(False) == 1 :
                            tmp_list = []
                            for el, el2 in zip(lst, current_options) :
                                if el :
                                    tmp_list.append(1+var_map.index(el2))
                                else :
                                    tmp_list.append((1+var_map.index(el2))*(-1))

                            # do not put duplicate elements in list
                            tmp_list.sort()
                            is_in_list = False
                            for item in clauses_2D:
                                if item == tmp_list:
                                    is_in_list = True
                            if not is_in_list :
                                clauses_2D.append(tmp_list)

            current_options = []

    p_line += str(len(var_map))+" "+str(len(clauses_2D))+"\n"

    for line in clauses_2D :
        for item in line :
            clauses += str(item)+" "
        clauses += "0\n"

    # und zack fertig .... dimacs
    final_string = variables_string+p_line+clauses
    return final_string.split('\n')


class Wrapper :
    
    def __init__(self, path_to_folder, project_name) :
        from os import listdir
        from os.path import isfile, join
        
        
        self.fitness_model = Fitness_Modell()
        self.constraint_model = Constraint_Model()
        
        self.best = []
        self.popsize = 100
        self.population = []
        self.fittness_list = []
       
        files = [join(path_to_folder, f) for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
        tmp_dimacs = []
        
        for file in files:
             # -------- Fitness Model -----------
            if file.count(project_name) > 0 and (file.count('feature') > 0 or file.count('interactions') > 0) :
                with open(file, 'r') as f:
                    self.fitness_model.add_fitness_rules(f.readlines())
            
             # ------ Constraint Model ---------
            if file.count(project_name) > 0 and file.count('.xml') > 0 :
                tmp_dimacs = parse_xml_to_dimacs(file)
                self.constraint_model.build_model(tmp_dimacs)
            if file.count(project_name) > 0 and file.count('.dimacs') > 0 :
                with open(file, 'r') as f:
                    tmp_dimacs = f.readlines()
                    self.constraint_model.build_model(tmp_dimacs)                    
                    
                                    
        # analyze constraints
        self.constraint_model.build_global_tabu_list()
        
        # parse fittnes rule names to ints
        self.fitness_model.parse_names_to_index(tmp_dimacs)

            
    def init_population(self) :
        random.seed()
        for i in range(self.popsize) :
            self.population.append(self.init_individual())
            
        tmp_list = []
        for it in self.population :
            if not it == None :
                tmp_list.append(it)
                
        self.population = tmp_list
        self.popsize = len(self.population)
        
        for i in range(0, self.popsize) :
            self.fittness_list.append(self.fitness_model.calculate_fitness(self.population[i]))
        
    def init_population_pycosat(self) :
        self.population = pycosat_init(self.constraint_model, self.popsize)
            
            
    def do_steady_state(self, max_rounds = 2000) :
        alg = Algorithms()
        alg.steady_state(self, self.popsize, max_rounds)
        
        
    def do_evolution(self, iterations = 100, u = 50, l = 200) :
        alg = Algorithms()
        alg.evolution(self, iterations, u, l)
        
        
    def init_individual(self) :     
        configuration = copy.deepcopy(self.constraint_model.mandatory_activation)
        tabu_config = copy.deepcopy(self.constraint_model.global_tabu_list)
        constraint_list = copy.deepcopy(self.constraint_model.constraint_list)
        
        is_valid = False
        
        counter = 0
        
        while not is_valid :
            for idx in range(len(configuration)) :
                if not tabu_config[idx]:
                    configuration[idx] = bool(random.getrandbits(1))
                    
            tmp = self.constraint_model.get_violated_variables(configuration)
            ptr = 0
            for it in tmp :
                for i in range(ptr + 1, it - 1) :
                    tabu_config[i] = True
                tabu_config[it] = False
                ptr = it
            
            counter += 1
            
            if self.constraint_model.check_validity(configuration)  :
                is_valid = True
                print (counter)
                return configuration
            if counter > 10000 : 
                print (counter)
                return None
            
        return None


def pycosat_init(constraint_model, number_of_individuals) :
    import pycosat
    import itertools
    
    cnf = constraint_model.get_constraints()
    pop_list = []
    tmp_individual = []
    
    
    population = list(itertools.islice(pycosat.itersolve(cnf), number_of_individuals*2))
    for individual in population :
            
        # represents the root element
        tmp_individual.append(True)

        for element in individual :
            if (element > 0) :
                tmp_individual.append(True)
            else :
                tmp_individual.append(False)
        pop_list.append(tmp_individual)
        tmp_individual = []
    
    print("pycosat done")
    return pop_list


class Algorithms :
    def __init__(self) :
        print("Initialize Algorithms Class")
        
        self.population = []
        self.fittness_of_population = []
        self.best = []        
        self.helper = Helper_Functions()
     
    
    # genetic
    def steady_state(self, wrapper, popsize, max_rounds) :
        import random
        random.seed()
        print("Steady State")
        
        #initialization
        max_mutations = 0.02
        self.population = []
        population_tmp = pycosat_init(wrapper.constraint_model, popsize)
        
        # appends elements, that are not already in p and assess fittness
        for ind in population_tmp :
            if not ind in self.population :
                # the pycosat individuals are very close, so we make big mutations in the begining
                ind = self.helper.mutate(ind, wrapper.constraint_model, 1)
                self.population.append(Individual(ind, wrapper.fitness_model.calculate_fitness(ind))) 

        # set popsize to the actual size of the population(in case not enough individuals could be produced)
        self.popsize = len(self.population)
        
        # set best
        self.best = max(self.population)
        
        # main algorithm
        not_done = True
        counter = 0
        children = []
        
        while not_done :            
            # select two parents
            parent_a = self.tournament_selection()
            parent_b = self.tournament_selection()
            
            # does not produce every time a valid child - crossover two parents
            children = self.helper.crossover(parent_a.activation_list, parent_b.activation_list, wrapper.constraint_model)
            
            # child a
            tmp = self.helper.mutate(children[0], wrapper.constraint_model, max_mutations)                
            tmp = Individual(tmp, wrapper.fitness_model.calculate_fitness(tmp))
            if tmp > self.best :
                self.best = tmp
                print("Child a is better than best at: " + str(self.best.fitness) + " / round: " + str(counter))
            self.population.append(tmp)
                
            # child b
            tmp = self.helper.mutate(children[1], wrapper.constraint_model, max_mutations)
            tmp = Individual(tmp, wrapper.fitness_model.calculate_fitness(tmp))

            children[1] = wrapper.fitness_model.calculate_fitness(children[1])
            if tmp > self.best :
                self.best = tmp
                print("Child b is better than best at: " + str(self.best.fitness) + " / round: " + str(counter))

            self.population.append(tmp)
            
            # exploit: kill 2 worst individuals
            self.population.sort()
            del(self.population[:2])
            
            counter += 1
            # maybe: or Best is optimum ?
            if counter > max_rounds :
                not_done = False
        
        
        print("Best: " + str(self.best))
        print("best cost:  " + str(wrapper.fitness_model.calculate_cost(self.best.activation_list)))
        
        
        
    def tournament_selection(self, tournament_size = 2) :
        tmp_last = -1
        tmp_pop = []
        for i in range(0, tournament_size) : 
            tmp_ind = random.randint(0, self.popsize-1)
            while(tmp_ind == tmp_last) :
                tmp_ind = random.randint(0, self.popsize-1)
            tmp_pop.append(self.population[tmp_ind])
            tmp_last = tmp_ind
                
        best = tmp_pop[0]
        for i in range(1, len(tmp_pop)) :
            if tmp_pop[i] > best :
                best = tmp_pop[i]
        
        return best
    
            
    def evolution(self, wrapper, iterations, u, l) :
        import random
        random.seed()
        print("EA")
        
        #initialization
        self.population = []
        max_mutations = 0.05
        population_tmp = pycosat_init(wrapper.constraint_model, l)
        
        # appends elements, that are not already in p and assess fittness
        for ind in population_tmp :
            if not ind in self.population :
                # the pycosat individuals are very close, so we make big mutations in the begining
                ind = self.helper.mutate(ind, wrapper.constraint_model, 1)
                self.population.append(Individual(ind, wrapper.fitness_model.calculate_fitness(ind)))        

        # sort population
        self.population.sort()
        
        # initialize best
        self.best = max(self.population)
        print("initial best: " + str(self.best.fitness))
        
        print("initialisation done!")
        
        
        # main algorithm
        not_done = True
        counter = 0 
        
        
        while not_done :
            del(self.population[:len(self.population)-u])   # truncate population
            popsize = len(self.population)
            
            # make l children
            popsize = len(self.population)
            for i in range(0, l) :
                tmp_child = self.helper.mutate(self.population[i % popsize].activation_list, wrapper.constraint_model, max_mutations)
                self.population.append(Individual(tmp_child, wrapper.fitness_model.calculate_fitness(tmp_child)))
            
            # sort population
            self.population.sort()
            
            if self.population[len(self.population) -1] > self.best :
                self.best = self.population[len(self.population) -1]
                print("new best found at iteration: " + str(counter) + "  Fitness = " + str(self.best.fitness))
            #print("last: " + str(self.population[len(self.population) -1]) + "  \  first: " + str(self.population[0]))
            
            counter += 1  
            if counter >= iterations :
                print(str(counter) + " iterations done")
                not_done = False 
                
        print("total best: " + str(self.best.fitness))
        print("best cost:  " + str(wrapper.fitness_model.calculate_cost(self.best.activation_list)))



class Helper_Functions :

    # This class contains needed functions to construct an machine learning algorithm
    def __init__(self) :
    	self.mutations = 0
        
    # TODO: to slow, more "randomness" needed
    def correct_violations(self, tweaked_individual, violated_constraint, constraint_model, tabu) :
        for it in violated_constraint :
            if not tabu[abs(it)] :
                tweaked_individual[abs(it)] = not tweaked_individual[abs(it)]
                tabu[abs(it)] = True
                violated_constraint_new = constraint_model.get_violated_constraint(tweaked_individual)
                if not violated_constraint_new == [] :
                    if not self.correct_violations(tweaked_individual, violated_constraint_new, constraint_model, tabu) :
                        tweaked_individual[abs(it)] = not tweaked_individual[abs(it)]
                    else :
                        self.mutations += 1
                        return True
                else :
                    self.mutations += 1
                    return True
        return False
      
    # TODO: to slow, more "randomness" needed
    # tries to tweak an individual by randomly modify its elements 
    # (we want to stay in the valid area, so check constraints)
    def mutate(self, individual, constraint_model, max_mutations) :
        import copy
        import random
        tweaked_individual = copy.deepcopy(individual)
        tabu = copy.deepcopy(constraint_model.global_tabu_list)
        
        random.seed()
        
        is_valid = False
        counter = 0
        len_individual = len(individual) 
        max_mutations = random.randint(0, int(max(max_mutations * len_individual, 0.5)))
        already_mutated = []
        
        counter = 0
        self.mutations = 0
        while self.mutations < max_mutations :
            gen_to_mutate = random.randint(1, len_individual - 1)
            
            # 50 percent chance to mutate thin gene
            if random.getrandbits(1) :
                while tabu[gen_to_mutate] : 
                    gen_to_mutate = random.randint(1, len_individual - 1)
                    counter += 1
                    if counter > 1000 :
                        #print("Tweak done at" + str(self.mutations) + "mutations")
                        return tweaked_individual

                tweaked_individual[gen_to_mutate] = not tweaked_individual[gen_to_mutate]
                tabu [gen_to_mutate] = True

                violated_constraint = constraint_model.get_violated_constraint(tweaked_individual)
                if not violated_constraint == [] :
                    if not self.correct_violations(tweaked_individual, violated_constraint, constraint_model, tabu) :
                        tweaked_individual[gen_to_mutate] = not tweaked_individual[gen_to_mutate] 
                    else :
                        self.mutations += 1      
            else :
                self.mutations += 1      
            
        return tweaked_individual
    
    
    def quality(self, individual, fitness_model) :
        # calgulates quality of an individual
        
        return fittness_model.calculate_fitness(individual)
        
    # TODO: to slow, more "randomness" needed
    def crossover(self, parent1, parent2, constraint_model) :
        # generates two children by crossovering two parents
        children = []
        children.append(copy.deepcopy(parent1))
        children.append(copy.deepcopy(parent2))
        tabu1 = copy.deepcopy(constraint_model.global_tabu_list)
        tabu2 = copy.deepcopy(constraint_model.global_tabu_list)
        
        for i in range(1, len(children[0])) :
            if (not children[0][i] == children[1][i]) and bool(random.getrandbits(1)) :
                if not(tabu1[i]) :
                    tabu1[i] = True
                    children[0][i] = not children[0][i]
                    violated_constraint = constraint_model.get_violated_constraint(children[0])
                    if not violated_constraint == [] :
                        if not self.correct_violations(children[0], violated_constraint, constraint_model, tabu1) :
                            children[0][i] = not children[0][i]
                if not(tabu2[i]) :
                    tabu2[i] = True
                    children[1][i] = not children[1][i]
                    violated_constraint = constraint_model.get_violated_constraint(children[1])
                    if not violated_constraint == [] :
                        if not self.correct_violations(children[1], violated_constraint, constraint_model, tabu2) :
                            children[1][i] = not children[1][i]
        
        return children
    
    
    def diversity(self, individual1, individual2) :
        # compares equality of two individuals element by element and return weighted sum
        
        if len(individual1) is not len(individual2) :
            print("size of both individuals should be equal to compare")
            return
        
        elements_equal = 0
        for el1, el2 in zip(individual1, individual2) :
            if el1 == el2 :
                elements_equal += 1

        return elements_equal/len(individual1)
    
    
    # TO-DO: implement!!
    def select_for_death (self, individual, population) :

        # select one element of the population and replace it with the individual
    	new_population = copy.deepcopy(population)
    	return new_population

if __name__ == '__main__':
	# testing import stuff
    import sys
    arguments = sys.argv[1:]
    #print(arguments)


    # single 
    if "h264" in arguments[0]:
        testinstanz = Wrapper('./project_public_1/', 'h264')
    elif "bdbc" in arguments[0]:
        testinstanz = Wrapper('./project_public_1/', 'bdbc')
    elif "busy" in arguments[0]:
        testinstanz = Wrapper('./project_public_2/', 'busy')
    elif "toy" in arguments[0]:
        testinstanz = MO_Wrapper('./project_public_2/', 'toy', ['feature1', 'feature2', 'feature3'], ['interactions1', 'interactions2', 'interactions3'])
        testinstanz.population.learn(100,100,500)
    else:
        print("could not load project")
        print("Usage: optimize.py model.xml model_feature.txt model_interactions.txt")

    #testinstanz.do_steady_state(max_rounds = 4000)
    testinstanz.do_evolution(iterations = 100, u = 20, l = 100)

        
